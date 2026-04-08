// constitution-search demonstrates HNSW approximate nearest neighbor search
// using the full text of three national constitutions (US, Canada, Mexico).
//
// Usage:
//
//	go run ./example/constitution-search "right to bear arms"
//	go run ./example/constitution-search -k 5 "freedom of speech"
//	go run ./example/constitution-search -save index.bin "judicial power"
//	go run ./example/constitution-search -load index.bin "due process"
package main

import (
	"flag"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/coder/hnsw"
)

const (
	embeddingDim  = 2048
	chunkMaxChars = 600
)

func main() {
	k := flag.Int("k", 3, "number of results to return")
	savePath := flag.String("save", "", "save index to file after building")
	loadPath := flag.String("load", "", "load index from file instead of building")
	dataDir := flag.String("data", "test-data", "directory containing .md constitution files")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: constitution-search [flags] <query>\n")
		flag.PrintDefaults()
		os.Exit(1)
	}
	query := strings.Join(flag.Args(), " ")

	var g *hnsw.Graph[string]
	var chunks map[string]string // key -> original text
	var idf *idfTable

	if *loadPath != "" {
		fmt.Printf("Loading index from %s...\n", *loadPath)
		sg, err := hnsw.LoadSavedGraph[string](*loadPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading index: %v\n", err)
			os.Exit(1)
		}
		g = sg.Graph
		chunks = make(map[string]string)
		// Build IDF from data dir even when loading index (needed for query embedding)
		idf, _ = buildIDF(*dataDir)
		fmt.Printf("Loaded %d vectors\n", g.Len())
	} else {
		var err error
		g, chunks, idf, err = buildIndex(*dataDir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error building index: %v\n", err)
			os.Exit(1)
		}
	}

	if *savePath != "" {
		fmt.Printf("Saving index to %s...\n", *savePath)
		sg := &hnsw.SavedGraph[string]{Graph: g, Path: *savePath}
		if err := sg.Save(); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving index: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("Saved.")
	}

	// Search
	queryVec := tfidfEmbedding(query, embeddingDim, idf)
	start := time.Now()
	results := g.SearchWithDistance(queryVec, *k)
	elapsed := time.Since(start)

	fmt.Printf("\nQuery: %q\n", query)
	fmt.Printf("Found %d results in %v\n\n", len(results), elapsed)

	for i, r := range results {
		fmt.Printf("--- Result %d (distance: %.4f) ---\n", i+1, r.Distance)
		fmt.Printf("Key: %s\n", r.Key)
		if text, ok := chunks[r.Key]; ok {
			// Show first 200 chars of the chunk
			preview := text
			if len(preview) > 200 {
				preview = preview[:200] + "..."
			}
			fmt.Printf("Text: %s\n", preview)
		}
		fmt.Println()
	}
}

// buildIDF reads all .md files in dataDir, chunks them, and returns an IDF table.
func buildIDF(dataDir string) (*idfTable, error) {
	files, err := filepath.Glob(filepath.Join(dataDir, "*.md"))
	if err != nil {
		return nil, err
	}
	var allTexts []string
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		allTexts = append(allTexts, chunkText(string(data), chunkMaxChars)...)
	}
	return computeIDF(allTexts), nil
}

func buildIndex(dataDir string) (*hnsw.Graph[string], map[string]string, *idfTable, error) {
	files, err := filepath.Glob(filepath.Join(dataDir, "*.md"))
	if err != nil {
		return nil, nil, nil, err
	}
	if len(files) == 0 {
		return nil, nil, nil, fmt.Errorf("no .md files found in %s", dataDir)
	}

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
		if err != nil {
			return nil, nil, nil, err
		}
		docName := strings.TrimSuffix(filepath.Base(f), ".md")
		chunks := chunkText(string(data), chunkMaxChars)
		fmt.Printf("Chunking %s (%d chunks)...\n", docName, len(chunks))
		for i, chunk := range chunks {
			allChunks = append(allChunks, docChunk{docName, i, chunk})
			allTexts = append(allTexts, chunk)
		}
	}

	idf := computeIDF(allTexts)
	fmt.Printf("Built IDF table: %d terms from %d chunks\n", len(idf.idf), idf.numDocs)

	// Second pass: embed with TF-IDF weights and index
	g := &hnsw.Graph[string]{
		M:        16,
		Ml:       0.25,
		Distance: hnsw.CosineDistance,
		EfSearch:  100,
		Rng:      rand.New(rand.NewSource(42)),
	}

	chunks := make(map[string]string)
	for _, dc := range allChunks {
		key := fmt.Sprintf("%s:%d", dc.docName, dc.index)
		vec := tfidfEmbedding(dc.text, embeddingDim, idf)
		g.Add(hnsw.Node[string]{Key: key, Value: vec})
		chunks[key] = dc.text
	}

	al := hnsw.Analyzer[string]{Graph: g}
	fmt.Printf("Index built: %d chunks, %d layers, topography: %v\n",
		len(allChunks), al.Height(), al.Topography())

	return g, chunks, idf, nil
}

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

func tokenize(text string) []string {
	raw := strings.Fields(strings.ToLower(text))
	out := make([]string, 0, len(raw))
	for _, w := range raw {
		w = strings.Trim(w, ".,;:!?\"'()[]{}*#>_\\-/")
		if len(w) < 2 || stopWords[w] {
			continue
		}
		out = append(out, w)
	}
	return out
}

type idfTable struct {
	idf      map[string]float32
	numDocs  int
	defaultW float32
}

func computeIDF(chunks []string) *idfTable {
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
		defaultW: float32(math.Log(n)),
	}
}

func (t *idfTable) weight(word string) float32 {
	if w, ok := t.idf[word]; ok {
		return w
	}
	return t.defaultW
}

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

// tfidfEmbedding produces a TF-IDF weighted feature-hashed embedding.
// Stop words are removed; each remaining word hashes to a dimension index,
// weighted by IDF. Shared words create direct cosine similarity signal.
func tfidfEmbedding(text string, dims int, idf *idfTable) hnsw.Vector {
	vec := make(hnsw.Vector, dims)
	words := tokenize(text)
	if len(words) == 0 {
		return vec
	}
	for _, word := range words {
		w := idf.weight(word)
		idx, sign := wordHash(word, dims)
		vec[idx] += sign * w
	}
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
