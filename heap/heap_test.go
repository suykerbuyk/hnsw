package heap

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

type Int int

func (i Int) Less(j Int) bool {
	return i < j
}

func TestHeap(t *testing.T) {
	h := Heap[Int]{}

	for i := 0; i < 20; i++ {
		h.Push(Int(rand.Int() % 100))
	}

	require.Equal(t, 20, h.Len())

	var inOrder []Int
	for h.Len() > 0 {
		inOrder = append(inOrder, h.Pop())
	}

	if !slices.IsSorted(inOrder) {
		t.Errorf("Heap did not return sorted elements: %+v", inOrder)
	}
}

func TestHeap_Max(t *testing.T) {
	h := Heap[Int]{}
	values := []Int{5, 1, 9, 3, 7, 2, 8, 4, 6}
	for _, v := range values {
		h.Push(v)
	}

	require.Equal(t, Int(9), h.Max(), "Max should return the largest element")
	require.Equal(t, Int(1), h.Min(), "Min should return the smallest element")
}

func TestHeap_Max_SingleElement(t *testing.T) {
	h := Heap[Int]{}
	h.Push(Int(42))
	require.Equal(t, Int(42), h.Max())
	require.Equal(t, Int(42), h.Min())
}

func TestHeap_Max_TwoElements(t *testing.T) {
	h := Heap[Int]{}
	h.Push(Int(10))
	h.Push(Int(20))
	require.Equal(t, Int(20), h.Max())
	require.Equal(t, Int(10), h.Min())
}

func TestHeap_Max_DuplicateValues(t *testing.T) {
	h := Heap[Int]{}
	for i := 0; i < 5; i++ {
		h.Push(Int(7))
	}
	require.Equal(t, Int(7), h.Max())
	require.Equal(t, Int(7), h.Min())
}

func TestHeap_PopLast(t *testing.T) {
	h := Heap[Int]{}
	values := []Int{5, 1, 9, 3, 7, 2, 8, 4, 6}
	for _, v := range values {
		h.Push(v)
	}

	// PopLast should remove and return the maximum
	got := h.PopLast()
	require.Equal(t, Int(9), got)
	require.Equal(t, 8, h.Len())

	// Next max should be 8
	got = h.PopLast()
	require.Equal(t, Int(8), got)
	require.Equal(t, 7, h.Len())

	// Drain via PopLast — should come out in descending order
	var descending []Int
	descending = append(descending, got)
	for h.Len() > 0 {
		descending = append(descending, h.PopLast())
	}
	for i := 1; i < len(descending); i++ {
		require.GreaterOrEqual(t, int(descending[i-1]), int(descending[i]),
			"PopLast should return elements in descending order")
	}
}

func TestHeap_PopLast_PreservesHeapInvariant(t *testing.T) {
	h := Heap[Int]{}
	for i := 0; i < 20; i++ {
		h.Push(Int(rand.Intn(100)))
	}

	// Remove the max, then verify Pop still returns sorted
	h.PopLast()

	var inOrder []Int
	for h.Len() > 0 {
		inOrder = append(inOrder, h.Pop())
	}
	require.True(t, slices.IsSorted(inOrder), "Heap invariant broken after PopLast: %v", inOrder)
}

func TestHeap_Init(t *testing.T) {
	data := []Int{5, 1, 9, 3, 7}
	h := Heap[Int]{}
	h.Init(data)

	require.Equal(t, 5, h.Len())
	require.Equal(t, Int(1), h.Min())
	require.Equal(t, Int(9), h.Max())

	var inOrder []Int
	for h.Len() > 0 {
		inOrder = append(inOrder, h.Pop())
	}
	require.True(t, slices.IsSorted(inOrder))
}

func TestHeap_Remove(t *testing.T) {
	h := Heap[Int]{}
	values := []Int{5, 1, 9, 3, 7}
	for _, v := range values {
		h.Push(v)
	}

	// Remove element at index 0 (the min)
	got := h.Remove(0)
	require.Equal(t, Int(1), got)
	require.Equal(t, 4, h.Len())

	// Remaining should still be a valid heap
	var inOrder []Int
	for h.Len() > 0 {
		inOrder = append(inOrder, h.Pop())
	}
	require.True(t, slices.IsSorted(inOrder))
	require.Equal(t, []Int{3, 5, 7, 9}, inOrder)
}

func TestHeap_Slice(t *testing.T) {
	h := Heap[Int]{}
	h.Push(Int(3))
	h.Push(Int(1))
	h.Push(Int(2))

	s := h.Slice()
	require.Equal(t, 3, len(s))
	// First element should be the min (heap property)
	require.Equal(t, Int(1), s[0])
}

func TestHeap_MaxCorrectnessStress(t *testing.T) {
	// Verify Max() is correct across many random configurations
	for trial := 0; trial < 100; trial++ {
		h := Heap[Int]{}
		n := rand.Intn(20) + 1
		var maxVal Int
		for i := 0; i < n; i++ {
			v := Int(rand.Intn(1000))
			h.Push(v)
			if i == 0 || v > maxVal {
				maxVal = v
			}
		}
		require.Equal(t, maxVal, h.Max(), "trial %d: Max incorrect for heap of size %d", trial, n)
	}
}
