// This file is a Go translation of src/lstm/lstmrecognizer.cpp and
// src/dict/dict.cpp from Tesseract OCR
// (https://github.com/tesseract-ocr/tesseract), licensed under the Apache
// License, Version 2.0. The translation is not verbatim.

package tessdata

import "fmt"

// Model is a fully loaded .traineddata: everything the LSTM recognition path
// reads, parsed into memory with every weight value present.
type Model struct {
	Version    string
	Swapped    bool
	Recognizer *Recognizer
	Unicharset *Unicharset
	Recoder    *Recoder

	// The three lexicons Dict::LoadLSTM reads (src/dict/dict.cpp:292). nil when
	// the component is absent; a lexicon-free model still recognizes text, it
	// just loses the dictionary-weighted beam.
	PuncDawg   *Dawg
	WordDawg   *Dawg
	NumberDawg *Dawg
}

// LoadModel parses a .traineddata file in Tesseract's native container layout
// and validates the invariants that bind its components together.
//
// The LSTM path requires TESSDATA_LSTM (17), TESSDATA_LSTM_UNICHARSET (21) and
// TESSDATA_LSTM_RECODER (22). When 21 and 22 are both present,
// LSTMRecognizer::DeSerialize reads the charsets from the components rather
// than from inside the LSTM blob; the embedded-charset layout is a pre-4.0
// arrangement no stock model uses, and cadmus does not implement it — a model
// missing either component is rejected rather than silently mis-parsed.
func LoadModel(data []byte) (*Model, error) {
	c, err := ParseContainer(data)
	if err != nil {
		return nil, fmt.Errorf("tessdata: parsing container: %w", err)
	}
	m := &Model{Version: c.Version(), Swapped: c.Swapped()}

	required := func(t Type) ([]byte, error) {
		b, ok := c.Entry(t)
		if !ok {
			return nil, fmt.Errorf("tessdata: model has no %v component", t)
		}
		return b, nil
	}

	lstm, err := required(TypeLSTM)
	if err != nil {
		return nil, err
	}
	if m.Recognizer, err = ParseRecognizer(lstm, c.Swapped()); err != nil {
		return nil, fmt.Errorf("tessdata: lstm component: %w", err)
	}

	ucs, err := required(TypeLSTMUnicharset)
	if err != nil {
		return nil, err
	}
	if m.Unicharset, err = ParseUnicharset(ucs); err != nil {
		return nil, fmt.Errorf("tessdata: lstm-unicharset component: %w", err)
	}

	rec, err := required(TypeLSTMRecoder)
	if err != nil {
		return nil, err
	}
	if m.Recoder, err = ParseRecoder(rec, c.Swapped()); err != nil {
		return nil, fmt.Errorf("tessdata: lstm-recoder component: %w", err)
	}

	for _, l := range []struct {
		typ Type
		dst **Dawg
	}{
		{TypeLSTMPuncDawg, &m.PuncDawg},
		{TypeLSTMSystemDawg, &m.WordDawg},
		{TypeLSTMNumberDawg, &m.NumberDawg},
	} {
		b, ok := c.Entry(l.typ)
		if !ok {
			continue
		}
		d, err := ParseDawg(b, c.Swapped())
		if err != nil {
			return nil, fmt.Errorf("tessdata: %v component: %w", l.typ, err)
		}
		*l.dst = d
	}

	if err := m.validate(); err != nil {
		return nil, err
	}
	return m, nil
}

// validate enforces the cross-component invariants. Each one is free to check
// and each one catches a whole class of mis-parse.
func (m *Model) validate() error {
	if m.Recoder.Size() != m.Unicharset.Size() {
		return fmt.Errorf("tessdata: recoder has %d entries but the unicharset has %d; they are 1:1", m.Recoder.Size(), m.Unicharset.Size())
	}

	sm := m.Softmax()
	if sm == nil {
		return fmt.Errorf("tessdata: network has no softmax output layer")
	}
	// LSTMTrainer::InitNetwork builds the network with recoder_.code_range() as
	// its output count, so these must agree exactly.
	if sm.NumOutputs != m.Recoder.CodeRange() {
		return fmt.Errorf("tessdata: softmax layer has %d outputs but the recoder's code range is %d", sm.NumOutputs, m.Recoder.CodeRange())
	}
	if m.Recognizer.Network.NumOutputs != sm.NumOutputs {
		return fmt.Errorf("tessdata: root layer declares %d outputs but the softmax layer has %d", m.Recognizer.Network.NumOutputs, sm.NumOutputs)
	}

	// UnicharCompress::DefragmentCodeValues deliberately relocates the null
	// code to the top of the range, so null_char_ is the LAST output index.
	// null_char_ is the authority; if this ever fires on a real model, prefer
	// the field and relax the check rather than deriving the blank from the
	// output count.
	if int(m.Recognizer.NullChar) != m.Recoder.CodeRange()-1 {
		return fmt.Errorf("tessdata: null_char is %d but the code range is %d; expected the blank at the last output index", m.Recognizer.NullChar, m.Recoder.CodeRange())
	}

	for _, d := range []*Dawg{m.PuncDawg, m.WordDawg, m.NumberDawg} {
		if d != nil && d.UnicharsetSize != m.Unicharset.Size() {
			return fmt.Errorf("tessdata: a dawg was built for a %d-entry unicharset but the model's has %d", d.UnicharsetSize, m.Unicharset.Size())
		}
	}
	return nil
}

// Softmax returns the network's unique softmax output layer, or nil if the
// graph does not have exactly one. Its NumOutputs is the authoritative output
// count — never the number in the spec string, which ParseOutput overrides.
//
// The count is deliberate, and n == 2 is reachable on a real (non-eng) model:
// NT_LSTM_SOFTMAX / NT_LSTM_SOFTMAX_ENCODED nest a FullyConnected softmax
// inside the LSTM's Children, so such a model has the nested one plus the
// network's own output layer. Returning nil there — and validate() therefore
// rejecting the model — is the correct L1a behaviour: cadmus does not
// implement that topology, and failing loudly beats guessing. Do not "fix"
// this into returning the outermost match.
func (m *Model) Softmax() *Layer {
	var found *Layer
	n := 0
	var walk func(*Layer)
	walk = func(l *Layer) {
		switch l.Type {
		case LayerSoftmax, LayerSoftmaxNoCTC:
			n++
			found = l
		}
		for _, c := range l.Children {
			walk(c)
		}
	}
	walk(m.Recognizer.Network)
	if n != 1 {
		return nil
	}
	return found
}

// NumOutputs is the network's output count, and therefore the width of one
// timestep of the network's output.
func (m *Model) NumOutputs() int { return m.Recognizer.Network.NumOutputs }

// NullChar is the network output index of the CTC blank.
func (m *Model) NullChar() int { return int(m.Recognizer.NullChar) }
