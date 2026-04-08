package hnsw

import (
	"bytes"
	"cmp"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_binaryVarint(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	i := 1337

	n, err := binaryWrite(buf, i)
	require.NoError(t, err)
	require.Equal(t, 2, n)

	// Ensure that binaryRead doesn't read past the
	// varint.
	buf.Write([]byte{0, 0, 0, 0})

	var j int
	_, err = binaryRead(buf, &j)
	require.NoError(t, err)
	require.Equal(t, 1337, j)

	require.Equal(
		t,
		[]byte{0, 0, 0, 0},
		buf.Bytes(),
	)
}

func Test_binaryWrite_string(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	s := "hello"

	n, err := binaryWrite(buf, s)
	require.NoError(t, err)
	// 5 bytes for the string, 1 byte for the length.
	require.Equal(t, 5+1, n)

	var s2 string
	_, err = binaryRead(buf, &s2)
	require.NoError(t, err)
	require.Equal(t, "hello", s2)

	require.Empty(t, buf.Bytes())
}

func verifyGraphNodes[K cmp.Ordered](t *testing.T, g *Graph[K]) {
	for _, layer := range g.layers {
		for _, node := range layer.nodes {
			for neighborKey, neighbor := range node.neighbors {
				_, ok := layer.nodes[neighbor.Key]
				if !ok {
					t.Errorf(
						"node %v has neighbor %v, but neighbor does not exist",
						node.Key, neighbor.Key,
					)
				}

				if neighborKey != neighbor.Key {
					t.Errorf("node %v has neighbor %v, but neighbor key is %v", node.Key,
						neighbor.Key,
						neighborKey,
					)
				}
			}
		}
	}
}

// requireGraphApproxEquals checks that two graphs are equal.
func requireGraphApproxEquals[K cmp.Ordered](t *testing.T, g1, g2 *Graph[K]) {
	require.Equal(t, g1.Len(), g2.Len())
	a1 := Analyzer[K]{g1}
	a2 := Analyzer[K]{g2}

	require.Equal(
		t,
		a1.Topography(),
		a2.Topography(),
	)

	require.Equal(
		t,
		a1.Connectivity(),
		a2.Connectivity(),
	)

	require.NotNil(t, g1.Distance)
	require.NotNil(t, g2.Distance)
	require.Equal(
		t,
		g1.Distance([]float32{0.5}, []float32{1}),
		g2.Distance([]float32{0.5}, []float32{1}),
	)

	require.Equal(t,
		g1.M,
		g2.M,
	)

	require.Equal(t,
		g1.Ml,
		g2.Ml,
	)

	require.Equal(t,
		g1.EfSearch,
		g2.EfSearch,
	)

	require.NotNil(t, g1.Rng)
	require.NotNil(t, g2.Rng)
}

func TestGraph_ExportImport(t *testing.T) {
	g1 := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g1.Add(
			Node[int]{
				i, randFloats(1),
			},
		)
	}

	buf := &bytes.Buffer{}
	err := g1.Export(buf)
	require.NoError(t, err)

	// Don't use newTestGraph to ensure parameters
	// are imported.
	g2 := &Graph[int]{}
	err = g2.Import(buf)
	require.NoError(t, err)

	requireGraphApproxEquals(t, g1, g2)

	n1 := g1.Search(
		[]float32{0.5},
		10,
	)

	n2 := g2.Search(
		[]float32{0.5},
		10,
	)

	require.Equal(t, n1, n2)

	verifyGraphNodes(t, g1)
	verifyGraphNodes(t, g2)
}

func TestSavedGraph(t *testing.T) {
	dir := t.TempDir()

	g1, err := LoadSavedGraph[int](dir + "/graph")
	require.NoError(t, err)
	require.Equal(t, 0, g1.Len())
	for i := 0; i < 128; i++ {
		g1.Add(
			Node[int]{
				i, randFloats(1),
			},
		)
	}

	err = g1.Save()
	require.NoError(t, err)

	g2, err := LoadSavedGraph[int](dir + "/graph")
	require.NoError(t, err)

	requireGraphApproxEquals(t, g1.Graph, g2.Graph)
}

func TestGraph_ExportUnregisteredDistance(t *testing.T) {
	g := newTestGraph[int]()
	g.Distance = func(a, b []float32) float32 { return 0 }
	g.Add(Node[int]{Key: 1, Value: Vector{1.0}})

	buf := &bytes.Buffer{}
	err := g.Export(buf)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must be registered")
}

func TestGraph_ImportUnknownDistance(t *testing.T) {
	// Build a valid export first, then tamper with the distance name
	g := newTestGraph[int]()
	g.Add(Node[int]{Key: 1, Value: Vector{1.0}})

	buf := &bytes.Buffer{}
	err := g.Export(buf)
	require.NoError(t, err)

	// Replace "euclidean" with "bogus_fn\x00\x00" in the binary
	data := buf.Bytes()
	dataStr := string(data)
	// Just import from a fresh buffer with bad distance
	badBuf := &bytes.Buffer{}
	// Write version, M, Ml, EfSearch, then a bad distance name
	_, err = multiBinaryWrite(badBuf, encodingVersion, 6, 0.5, 20, "nonexistent")
	require.NoError(t, err)
	_ = dataStr // suppress unused

	g2 := &Graph[int]{}
	err = g2.Import(badBuf)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unknown distance function")
}

func TestGraph_ImportBadVersion(t *testing.T) {
	buf := &bytes.Buffer{}
	_, err := multiBinaryWrite(buf, 999, 6, 0.5, 20, "euclidean")
	require.NoError(t, err)

	g := &Graph[int]{}
	err = g.Import(buf)
	require.Error(t, err)
	require.Contains(t, err.Error(), "incompatible encoding version")
}

func TestGraph_ImportTruncated(t *testing.T) {
	g := &Graph[int]{}
	buf := &bytes.Buffer{}
	err := g.Import(buf)
	require.Error(t, err)
}

func TestGraph_ExportImportStringKeys(t *testing.T) {
	g1 := &Graph[string]{
		M:        6,
		Distance: CosineDistance,
		Ml:       0.5,
		EfSearch: 20,
		Rng:      rand.New(rand.NewSource(0)),
	}
	g1.Add(
		Node[string]{Key: "hello", Value: Vector{1, 0}},
		Node[string]{Key: "world", Value: Vector{0, 1}},
	)

	buf := &bytes.Buffer{}
	err := g1.Export(buf)
	require.NoError(t, err)

	g2 := &Graph[string]{}
	err = g2.Import(buf)
	require.NoError(t, err)

	requireGraphApproxEquals(t, g1, g2)

	vec, ok := g2.Lookup("hello")
	require.True(t, ok)
	require.Equal(t, Vector{1, 0}, vec)
}

const benchGraphSize = 100

func BenchmarkGraph_Import(b *testing.B) {
	b.ReportAllocs()
	g := newTestGraph[int]()
	for i := 0; i < benchGraphSize; i++ {
		g.Add(
			Node[int]{
				i, randFloats(256),
			},
		)
	}

	buf := &bytes.Buffer{}
	err := g.Export(buf)
	require.NoError(b, err)

	b.ResetTimer()
	b.SetBytes(int64(buf.Len()))

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		rdr := bytes.NewReader(buf.Bytes())
		g := newTestGraph[int]()
		b.StartTimer()
		err = g.Import(rdr)
		require.NoError(b, err)
	}
}

func BenchmarkGraph_Export(b *testing.B) {
	b.ReportAllocs()
	g := newTestGraph[int]()
	for i := 0; i < benchGraphSize; i++ {
		g.Add(
			Node[int]{
				i, randFloats(256),
			},
		)
	}

	var buf bytes.Buffer
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := g.Export(&buf)
		require.NoError(b, err)
		if i == 0 {
			ln := buf.Len()
			b.SetBytes(int64(ln))
		}
		buf.Reset()
	}
}
