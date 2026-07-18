package nlp

import (
	"regexp"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

var wordRegex = regexp.MustCompile(`\b\p{L}{2,}\b`)

type CountVectorizer interface {
	Fit(train ...string)
	Transform(texts ...string) *mat.Dense
	FitTransform(texts ...string) *mat.Dense
	GetVocabulary() map[string]int
}

type countVectorizer struct {
	numWords  int
	wordIndex map[string]int
	stopWords map[string]bool
}

// encodes one or more text documents to obtain the frequency
// with which the corresponding term appears in the corresponding document
func NewCountVectorizer(numWords int, stopWords ...string) CountVectorizer {
	stop := make(map[string]bool)
	for _, word := range stopWords {
		stop[word] = true
	}

	return &countVectorizer{
		numWords:  numWords,
		wordIndex: make(map[string]int),
		stopWords: stop,
	}
}

// processes the supplied training data to populate wordIndex map
func (v *countVectorizer) Fit(texts ...string) {
	if len(v.wordIndex) != 0 {
		v.wordIndex = make(map[string]int)
	}

	// count term frequencies
	freq := make(map[string]int)
	for _, text := range texts {
		words := v.Tokenize(text)
		for _, word := range words {
			freq[word]++
		}
	}

	// convert map to slice
	type termFreq struct {
		term string
		freq int
	}
	terms := make([]termFreq, 0, len(freq))
	for term, f := range freq {
		terms = append(terms, termFreq{term, f})
	}

	// sort by frequency (descending)
	sort.Slice(terms, func(i, j int) bool {
		if terms[i].freq == terms[j].freq {
			return terms[i].term < terms[j].term
		}
		return terms[i].freq > terms[j].freq
	})

	// keep only numWords
	n := min(v.numWords, len(terms))
	for i := 0; i < n; i++ {
		v.wordIndex[terms[i].term] = i
	}
}

// transforms the given documents into the frequency with which the associated term occurs
func (v *countVectorizer) Transform(texts ...string) *mat.Dense {
	m := mat.NewDense(len(v.wordIndex), len(texts), nil)

	for j, text := range texts {
		words := v.Tokenize(text)
		for _, word := range words {
			if i, exists := v.wordIndex[word]; exists {
				m.Set(i, j, m.At(i, j)+1)
			}
		}
	}

	return m
}

// FitTransform is exactly equivalent to calling Fit() followed by Transform()
func (v *countVectorizer) FitTransform(texts ...string) *mat.Dense {
	v.Fit(texts...)
	return v.Transform(texts...)
}

// returns a slice of all the tokens contained in string text
func (v *countVectorizer) Tokenize(text string) []string {
	c := strings.ToLower(text)

	// match whole words, removing any punctuation/whitespace
	words := wordRegex.FindAllString(c, -1)

	// filter out stop words
	if v.stopWords != nil {
		b := words[:0]
		for _, w := range words {
			if !v.stopWords[w] {
				b = append(b, w)
			}
		}
		return b
	}

	return words
}

// return vocabulary map is populated by the Fit() or FitTransform() methods
func (v *countVectorizer) GetVocabulary() map[string]int {
	return v.wordIndex
}
