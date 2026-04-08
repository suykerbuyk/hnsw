package hnsw

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

const (
	embeddingDim  = 2048
	chunkMaxChars = 600
)

// chunkText splits text into chunks of roughly maxChars characters,
// breaking at paragraph boundaries (double newline).
func chunkText(text string, maxChars int) []string {
	paragraphs := strings.Split(text, "\n\n")
	var chunks []string
	var current strings.Builder

	for _, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if current.Len() > 0 && current.Len()+len(p)+2 > maxChars {
			chunks = append(chunks, current.String())
			current.Reset()
		}
		if current.Len() > 0 {
			current.WriteString("\n\n")
		}
		current.WriteString(p)
	}
	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}
	return chunks
}

// wordHash returns a (dimension index, sign) pair for a word.
// This is the "hashing trick" — each word deterministically maps to one
// dimension, so shared words between query and document create direct
// overlap in the vector, unlike random word vectors.
func wordHash(word string, dims int) (int, float32) {
	h := fnv.New64a()
	h.Write([]byte(word))
	hash := h.Sum64()
	idx := int(hash % uint64(dims))
	sign := float32(1.0)
	if hash&(1<<32) != 0 {
		sign = -1.0
	}
	return idx, sign
}

// stopWords are common English words that carry little discriminating value.
// Removing them lets rare, meaningful words dominate the embedding.
var stopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "by": true, "for": true, "from": true, "has": true, "he": true,
	"in": true, "is": true, "it": true, "its": true, "of": true, "on": true,
	"or": true, "that": true, "the": true, "to": true, "was": true, "were": true,
	"will": true, "with": true, "this": true, "but": true, "they": true,
	"have": true, "had": true, "not": true, "been": true, "no": true,
	"shall": true, "such": true, "any": true, "each": true, "which": true,
	"their": true, "if": true, "than": true, "other": true, "into": true,
	"may": true, "all": true, "who": true, "when": true, "upon": true,
}

// tokenize splits text into lowercase words with stop words removed.
func tokenize(text string) []string {
	raw := strings.Fields(strings.ToLower(text))
	out := make([]string, 0, len(raw))
	for _, w := range raw {
		// Strip common markdown/punctuation from edges
		w = strings.Trim(w, ".,;:!?\"'()[]{}*#>_\\-/")
		if len(w) < 2 || stopWords[w] {
			continue
		}
		out = append(out, w)
	}
	return out
}

// idfTable holds inverse document frequency weights computed from a corpus.
// Words that appear in many documents get low weight; rare words get high weight.
type idfTable struct {
	idf      map[string]float32
	numDocs  int
	defaultW float32 // weight for unseen words (max IDF)
}

// buildIDF computes IDF weights from a set of text chunks.
// IDF(word) = log(N / df(word)) where df is the number of chunks containing the word.
func buildIDF(chunks []string) *idfTable {
	df := make(map[string]int)
	for _, chunk := range chunks {
		seen := make(map[string]bool)
		for _, w := range tokenize(chunk) {
			if !seen[w] {
				df[w]++
				seen[w] = true
			}
		}
	}

	n := float64(len(chunks))
	idf := make(map[string]float32, len(df))
	for word, count := range df {
		idf[word] = float32(math.Log(n / float64(count)))
	}

	return &idfTable{
		idf:      idf,
		numDocs:  len(chunks),
		defaultW: float32(math.Log(n)), // max possible IDF for unseen words
	}
}

func (t *idfTable) weight(word string) float32 {
	if w, ok := t.idf[word]; ok {
		return w
	}
	return t.defaultW
}

// tfidfEmbedding produces a TF-IDF weighted feature-hashed embedding.
// Stop words are removed. Each remaining word hashes to a dimension index
// (the "hashing trick"), weighted by its IDF score. Shared words between
// query and document contribute to the SAME dimensions, creating direct
// cosine similarity signal — unlike random word vectors where shared words
// contribute to unrelated directions.
func tfidfEmbedding(text string, dims int, idf *idfTable) Vector {
	vec := make(Vector, dims)
	words := tokenize(text)
	if len(words) == 0 {
		return vec
	}

	for _, word := range words {
		w := idf.weight(word)
		idx, sign := wordHash(word, dims)
		vec[idx] += sign * w
	}

	// L2 normalize
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
	return vec
}

// bruteForceKNN returns the k nearest neighbors by exhaustive search.
func bruteForceKNN(query Vector, nodes []Node[string], k int, dist DistanceFunc) []Node[string] {
	type scored struct {
		node Node[string]
		dist float32
	}
	results := make([]scored, len(nodes))
	for i, n := range nodes {
		results[i] = scored{node: n, dist: dist(query, n.Value)}
	}
	sort.Slice(results, func(i, j int) bool {
		return results[i].dist < results[j].dist
	})
	if k > len(results) {
		k = len(results)
	}
	out := make([]Node[string], k)
	for i := 0; i < k; i++ {
		out[i] = results[i].node
	}
	return out
}

// recallAtK computes the fraction of ground-truth results found in predicted.
func recallAtK(predicted, groundTruth []Node[string]) float64 {
	truthSet := make(map[string]struct{}, len(groundTruth))
	for _, n := range groundTruth {
		truthSet[n.Key] = struct{}{}
	}
	var hits int
	for _, n := range predicted {
		if _, ok := truthSet[n.Key]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(groundTruth))
}

func loadTestDocuments(t *testing.T) []Node[string] {
	t.Helper()
	files, err := filepath.Glob("test-data/*.md")
	require.NoError(t, err)
	require.NotEmpty(t, files, "no test documents found in test-data/")

	// First pass: collect all chunks to build IDF table
	type docChunk struct {
		docName string
		index   int
		text    string
	}
	var allChunks []docChunk
	var allTexts []string
	for _, f := range files {
		data, err := os.ReadFile(f)
		require.NoError(t, err)
		docName := strings.TrimSuffix(filepath.Base(f), ".md")
		chunks := chunkText(string(data), chunkMaxChars)
		for i, chunk := range chunks {
			allChunks = append(allChunks, docChunk{docName, i, chunk})
			allTexts = append(allTexts, chunk)
		}
	}

	idf := buildIDF(allTexts)

	// Second pass: embed with TF-IDF weights
	var nodes []Node[string]
	for _, dc := range allChunks {
		key := fmt.Sprintf("%s:%d", dc.docName, dc.index)
		nodes = append(nodes, Node[string]{
			Key:   key,
			Value: tfidfEmbedding(dc.text, embeddingDim, idf),
		})
	}
	return nodes
}

func TestIntegration_ConstitutionSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	nodes := loadTestDocuments(t)
	require.Greater(t, len(nodes), 50, "expected substantial number of chunks")

	g := &Graph[string]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 100,
		Rng:      rand.New(rand.NewSource(42)),
	}

	for _, n := range nodes {
		g.Add(n)
	}
	require.Equal(t, len(nodes), g.Len())

	// Test recall against brute-force ground truth
	ks := []int{5, 10, 20}
	// Use a set of query nodes from the corpus itself
	queryIndices := []int{0, len(nodes) / 4, len(nodes) / 2, 3 * len(nodes) / 4, len(nodes) - 1}

	for _, k := range ks {
		var totalRecall float64
		var nQueries int
		for _, qi := range queryIndices {
			query := nodes[qi].Value
			hnswResults := g.Search(query, k)
			bfResults := bruteForceKNN(query, nodes, k, CosineDistance)
			recall := recallAtK(hnswResults, bfResults)
			totalRecall += recall
			nQueries++
		}
		avgRecall := totalRecall / float64(nQueries)
		t.Logf("recall@%d: %.2f (averaged over %d queries)", k, avgRecall, nQueries)
		require.GreaterOrEqual(t, avgRecall, 0.6,
			"recall@%d too low: %.2f", k, avgRecall)
	}
}

func TestIntegration_ConstitutionDeleteAndSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	nodes := loadTestDocuments(t)

	g := &Graph[string]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 100,
		Rng:      rand.New(rand.NewSource(42)),
	}

	for _, n := range nodes {
		g.Add(n)
	}

	preLen := g.Len()

	// Delete 20% of nodes
	rng := rand.New(rand.NewSource(99))
	deleteCount := preLen / 5
	perm := rng.Perm(len(nodes))
	deletedKeys := make(map[string]struct{})
	for i := 0; i < deleteCount; i++ {
		key := nodes[perm[i]].Key
		ok := g.Delete(key)
		require.True(t, ok)
		deletedKeys[key] = struct{}{}
	}

	require.Equal(t, preLen-deleteCount, g.Len())

	// Build surviving nodes list for brute-force
	var surviving []Node[string]
	for _, n := range nodes {
		if _, deleted := deletedKeys[n.Key]; !deleted {
			surviving = append(surviving, n)
		}
	}

	// Search should still work and find reasonable results
	query := surviving[0].Value
	hnswResults := g.Search(query, 10)
	bfResults := bruteForceKNN(query, surviving, 10, CosineDistance)
	recall := recallAtK(hnswResults, bfResults)
	t.Logf("recall@10 after 20%% deletion: %.2f", recall)
	require.GreaterOrEqual(t, recall, 0.5,
		"recall degraded too much after deletion")

	// Check graph connectivity
	al := Analyzer[string]{Graph: g}
	connectivity := al.Connectivity()
	require.NotEmpty(t, connectivity)
	t.Logf("post-delete connectivity: %v", connectivity)
	// Base layer should still have some connectivity
	require.Greater(t, connectivity[0], float64(0),
		"base layer lost all connectivity")
}

func TestIntegration_ConstitutionExportImport(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	nodes := loadTestDocuments(t)

	g := &Graph[string]{
		M:        16,
		Ml:       0.25,
		Distance: CosineDistance,
		EfSearch: 100,
		Rng:      rand.New(rand.NewSource(42)),
	}
	for _, n := range nodes {
		g.Add(n)
	}

	// Export and reimport
	dir := t.TempDir()
	sg := &SavedGraph[string]{Graph: g, Path: filepath.Join(dir, "constitution.idx")}
	err := sg.Save()
	require.NoError(t, err)

	sg2, err := LoadSavedGraph[string](filepath.Join(dir, "constitution.idx"))
	require.NoError(t, err)
	require.Equal(t, g.Len(), sg2.Len())

	// Search results should match between original and imported
	query := nodes[0].Value
	r1 := g.Search(query, 5)
	r2 := sg2.Search(query, 5)
	require.Equal(t, r1, r2)
}

func BenchmarkIntegration_Constitution(b *testing.B) {
	files, _ := filepath.Glob("test-data/*.md")
	if len(files) == 0 {
		b.Skip("no test data")
	}

	var allTexts []string
	type dc struct {
		name  string
		index int
		text  string
	}
	var allChunks []dc
	for _, f := range files {
		data, _ := os.ReadFile(f)
		docName := strings.TrimSuffix(filepath.Base(f), ".md")
		chunks := chunkText(string(data), chunkMaxChars)
		for i, chunk := range chunks {
			allChunks = append(allChunks, dc{docName, i, chunk})
			allTexts = append(allTexts, chunk)
		}
	}
	idf := buildIDF(allTexts)
	var allNodes []Node[string]
	for _, c := range allChunks {
		key := fmt.Sprintf("%s:%d", c.name, c.index)
		allNodes = append(allNodes, Node[string]{
			Key:   key,
			Value: tfidfEmbedding(c.text, embeddingDim, idf),
		})
	}

	b.Run("Insert", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			g := &Graph[string]{
				M: 16, Ml: 0.25, Distance: CosineDistance,
				EfSearch: 100, Rng: rand.New(rand.NewSource(42)),
			}
			for _, n := range allNodes {
				g.Add(n)
			}
		}
	})

	// Build once for search/delete benchmarks
	g := &Graph[string]{
		M: 16, Ml: 0.25, Distance: CosineDistance,
		EfSearch: 100, Rng: rand.New(rand.NewSource(42)),
	}
	for _, n := range allNodes {
		g.Add(n)
	}

	b.Run("Search", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			g.Search(allNodes[i%len(allNodes)].Value, 10)
		}
	})
}
