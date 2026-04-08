package hnsw

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEuclideanDistance(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	require.Equal(t, float32(5.196152), EuclideanDistance(a, b))
}

func TestCosineSimilarity(t *testing.T) {
	var a, b []float32
	// Same magnitude, same direction.
	a = []float32{1, 1, 1}
	b = []float32{0.8, 0.8, 0.8}
	require.InDelta(t, 0, CosineDistance(a, b), 0.000001)

	// Perpendicular vectors.
	a = []float32{1, 0}
	b = []float32{0, 1}
	require.InDelta(t, 1, CosineDistance(a, b), 0.000001)

	// Equivalent vectors.
	a = []float32{1, 0}
	b = []float32{1, 0}
	require.InDelta(t, 0, CosineDistance(a, b), 0.000001)
}

func TestRegisterDistanceFunc(t *testing.T) {
	custom := func(a, b []float32) float32 { return 42 }
	RegisterDistanceFunc("custom_test", custom)

	name, ok := distanceFuncToName(custom)
	require.True(t, ok)
	require.Equal(t, "custom_test", name)

	// Clean up
	delete(distanceFuncs, "custom_test")
}

func TestDistanceFuncToName(t *testing.T) {
	name, ok := distanceFuncToName(CosineDistance)
	require.True(t, ok)
	require.Equal(t, "cosine", name)

	name, ok = distanceFuncToName(EuclideanDistance)
	require.True(t, ok)
	require.Equal(t, "euclidean", name)
}

func TestDistanceFuncToName_Unknown(t *testing.T) {
	unknown := func(a, b []float32) float32 { return 0 }
	_, ok := distanceFuncToName(unknown)
	require.False(t, ok)
}

func BenchmarkEuclideanDistance(b *testing.B) {
	v1 := randFloats(1536)
	v2 := randFloats(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EuclideanDistance(v1, v2)
	}
}

func BenchmarkCosineSimilarity(b *testing.B) {
	v1 := randFloats(1536)
	v2 := randFloats(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineDistance(v1, v2)
	}
}
