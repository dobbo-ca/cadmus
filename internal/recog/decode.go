// This file is a Go translation of RecodeBeamSearch::ExtractBestPathAsLabels in
// src/lstm/recodebeam.cpp and LSTMRecognizer::DecodeLabels in
// src/lstm/lstmrecognizer.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package recog

import (
	"fmt"
	"image"
	"strings"

	"github.com/dobbo-ca/cadmus/internal/dict"
	"github.com/dobbo-ca/cadmus/internal/nn"
	"github.com/dobbo-ca/cadmus/internal/tessdata"
)

// Recognizer is a loaded model ready to transcribe line images.
type Recognizer struct {
	Net     *Network
	Charset *tessdata.Unicharset
	Recoder *tessdata.Recoder
	// Dict is the lexicon. Nil disables the dictionary beams; Recognize still
	// works, at greedy-equivalent accuracy.
	Dict *dict.Dict
	// lexicon is what the model actually carried, so SetLexicon(true) can put
	// Dict back after SetLexicon(false) cleared it.
	lexicon *dict.Dict
}

// SetLexicon enables or disables dictionary weighting. Disabling it makes the
// beam search behave like greedy decoding with better certainties, because the
// score merge stays but the dawg beams are never populated.
//
// L2's WithLexicon(bool) option calls this.
func (r *Recognizer) SetLexicon(enabled bool) {
	if enabled {
		r.Dict = r.lexicon
		return
	}
	r.Dict = nil
}

// NewRecognizer loads a .traineddata file.
func NewRecognizer(model []byte) (*Recognizer, error) {
	c, err := tessdata.ParseContainer(model)
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}
	lstm, ok := c.Entry(tessdata.TypeLSTM)
	if !ok {
		return nil, fmt.Errorf("recog: model has no lstm component")
	}
	rec, err := tessdata.ParseRecognizer(lstm, c.Swapped())
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}
	net, err := Build(rec)
	if err != nil {
		return nil, err
	}

	// LSTMRecognizer::LoadCharsets reads components 21 and 22 when both are
	// present, which is the case for every model in tessdata, tessdata_best and
	// tessdata_fast. The legacy embedded-charset layout is not supported.
	csEntry, ok := c.Entry(tessdata.TypeLSTMUnicharset)
	if !ok {
		return nil, fmt.Errorf("recog: model has no lstm-unicharset component; the embedded-charset layout is not supported")
	}
	cs, err := tessdata.ParseUnicharset(csEntry)
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}
	rcEntry, ok := c.Entry(tessdata.TypeLSTMRecoder)
	if !ok {
		return nil, fmt.Errorf("recog: model has no lstm-recoder component")
	}
	rc, err := tessdata.ParseRecoder(rcEntry, c.Swapped())
	if err != nil {
		return nil, fmt.Errorf("recog: %w", err)
	}

	// Free cross-checks between the four components. Each of these has a
	// distinct failure it catches, and all are cheap.
	if rc.CodeRange() != net.NumOutputs {
		return nil, fmt.Errorf("recog: recoder code range %d does not match the %d network outputs", rc.CodeRange(), net.NumOutputs)
	}
	if rc.Size() != cs.Size() {
		return nil, fmt.Errorf("recog: recoder has %d entries but the unicharset has %d", rc.Size(), cs.Size())
	}
	// L1b implements only the length-1 recoder fast path: network output index
	// -> code -> unichar id is then a flat near-permutation, and the whole
	// partial-code dimension of RecodeBeamSearch (kNumLengths, GetNextCodes,
	// GetFinalCodes) collapses to a no-op. internal/tessdata loads multi-code
	// recoders happily; the restriction is the beam's, so it lives here.
	if rc.MaxCodeLen() != 1 {
		return nil, fmt.Errorf("recog: recoder has codes up to %d long; L1b supports single-code recoders only (see cad-q16)", rc.MaxCodeLen())
	}
	if space := rc.Encode(tessdata.UnicharSpace); len(space) != 1 || space[0] != tessdata.UnicharSpace {
		return nil, fmt.Errorf("recog: space was garbled in recoding")
	}

	// Dict::LoadLSTM (src/dict/dict.cpp:292) reads the punctuation, word and
	// number dawgs. A model missing them still recognizes text; it just loses
	// the dictionary-weighted beam, so an absent component is not an error.
	var punc, word, number *tessdata.Dawg
	for _, l := range []struct {
		typ tessdata.Type
		dst **tessdata.Dawg
	}{
		{tessdata.TypeLSTMPuncDawg, &punc},
		{tessdata.TypeLSTMSystemDawg, &word},
		{tessdata.TypeLSTMNumberDawg, &number},
	} {
		e, ok := c.Entry(l.typ)
		if !ok {
			continue
		}
		d, err := tessdata.ParseDawg(e, c.Swapped())
		if err != nil {
			return nil, fmt.Errorf("recog: %v component: %w", l.typ, err)
		}
		if d.UnicharsetSize != cs.Size() {
			return nil, fmt.Errorf("recog: the %v was built for a %d-entry unicharset but the model's has %d", l.typ, d.UnicharsetSize, cs.Size())
		}
		*l.dst = d
	}
	var lex *dict.Dict
	if punc != nil || word != nil || number != nil {
		lex = dict.New(cs, punc, word, number)
	}
	return &Recognizer{Net: net, Charset: cs, Recoder: rc, Dict: lex, lexicon: lex}, nil
}

// Symbol is one decoded character and the timestep run it occupies.
type Symbol struct {
	UnicharID int
	Text      string
	Start     int
	End       int
}

// Forward normalizes a line image and runs it through the network, returning
// the softmax output and the normalization metadata the geometry needs.
func (r *Recognizer) Forward(img image.Image) (*nn.Tensor, *Normalized, error) {
	// Normalize shares the network's randomizer and must run before the graph,
	// exactly as Copy2DImage precedes Convolve::Forward.
	norm, err := Normalize(img, r.Net.InputHeight, r.Net.XScale, r.Net.Rand)
	if err != nil {
		return nil, nil, err
	}
	out, err := r.Net.Root.Forward(norm.Input)
	if err != nil {
		return nil, nil, fmt.Errorf("recog: forward: %w", err)
	}
	if out.Features != r.Net.NumOutputs {
		return nil, nil, fmt.Errorf("recog: network produced %d outputs, want %d", out.Features, r.Net.NumOutputs)
	}
	return out, norm, nil
}

// GreedyDecode is the CTC best-path decode: per-timestep argmax, blanks
// dropped, runs of the same code collapsed to one symbol.
//
// This is deliberately not LabelsViaSimpleText, which omits the run collapse
// and is only correct for models trained without CTC.
func (r *Recognizer) GreedyDecode(out *nn.Tensor) ([]Symbol, error) {
	var syms []Symbol
	prev := -1
	for t := range out.Map.Len() {
		code := argmax(out.Row(t))
		if code != prev && code != r.Net.NullChar {
			id, ok := r.Recoder.DecodeUnichar([]int32{int32(code)})
			if !ok || id < 0 || id >= r.Charset.Size() {
				return nil, fmt.Errorf("recog: code %d at t=%d does not decode to a unichar id inside the charset (got %d, ok=%v)", code, t, id, ok)
			}
			syms = append(syms, Symbol{UnicharID: id, Text: r.Charset.Text(id), Start: t, End: t + 1})
		} else if code == prev && len(syms) > 0 && code != r.Net.NullChar {
			syms[len(syms)-1].End = t + 1
		}
		prev = code
	}
	return syms, nil
}

// RecognizeText is the whole Stage 1 pipeline: image in, text out.
func (r *Recognizer) RecognizeText(img image.Image) (string, error) {
	out, _, err := r.Forward(img)
	if err != nil {
		return "", err
	}
	syms, err := r.GreedyDecode(out)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, s := range syms {
		b.WriteString(s.Text)
	}
	return b.String(), nil
}

func argmax(row []float32) int {
	best, bestI := row[0], 0
	for i, v := range row[1:] {
		if v > best {
			best, bestI = v, i+1
		}
	}
	return bestI
}
