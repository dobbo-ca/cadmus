package tessdata

import "testing"

func TestReaderPrimitivesLittleEndian(t *testing.T) {
	// uint32(1), int32(-2), int8(7)
	data := []byte{
		0x01, 0x00, 0x00, 0x00,
		0xfe, 0xff, 0xff, 0xff,
		0x07,
	}
	r := NewReader(data)

	if got, err := r.Uint32(); err != nil || got != 1 {
		t.Fatalf("Uint32() = %d, %v; want 1, nil", got, err)
	}
	if got, err := r.Int32(); err != nil || got != -2 {
		t.Fatalf("Int32() = %d, %v; want -2, nil", got, err)
	}
	if got, err := r.Int8(); err != nil || got != 7 {
		t.Fatalf("Int8() = %d, %v; want 7, nil", got, err)
	}
	if r.Remaining() != 0 {
		t.Fatalf("Remaining() = %d; want 0", r.Remaining())
	}
}

func TestReaderSwapReversesMultiByteReads(t *testing.T) {
	r := NewReader([]byte{0x00, 0x00, 0x00, 0x01})
	r.SetSwap(true)
	got, err := r.Uint32()
	if err != nil || got != 1 {
		t.Fatalf("Uint32() with swap = %d, %v; want 1, nil", got, err)
	}
}

// String is a uint32 length followed by that many raw bytes. Tesseract rejects
// lengths above 50,000,000 as corruption; so do we.
func TestReaderString(t *testing.T) {
	data := []byte{0x03, 0x00, 0x00, 0x00, 'a', 'b', 'c'}
	r := NewReader(data)
	got, err := r.String()
	if err != nil || got != "abc" {
		t.Fatalf("String() = %q, %v; want \"abc\", nil", got, err)
	}
}

func TestReaderStringRejectsAbsurdLength(t *testing.T) {
	data := []byte{0xff, 0xff, 0xff, 0xff}
	r := NewReader(data)
	if _, err := r.String(); err == nil {
		t.Fatal("String() with 4294967295-byte length: want error, got nil")
	}
}

func TestReaderTruncatedInputErrors(t *testing.T) {
	r := NewReader([]byte{0x01, 0x02})
	if _, err := r.Uint32(); err == nil {
		t.Fatal("Uint32() on 2-byte input: want error, got nil")
	}
}

func TestReaderInt16(t *testing.T) {
	// kDawgMagicNumber == 42, written as int16.
	r := NewReader([]byte{0x2a, 0x00, 0xff, 0xff})
	if got, err := r.Int16(); err != nil || got != 42 {
		t.Fatalf("Int16() = %d, %v; want 42, nil", got, err)
	}
	if got, err := r.Int16(); err != nil || got != -1 {
		t.Fatalf("Int16() = %d, %v; want -1, nil", got, err)
	}
	if r.Remaining() != 0 {
		t.Fatalf("Remaining() = %d; want 0", r.Remaining())
	}
}

func TestReaderInt16Swapped(t *testing.T) {
	r := NewReader([]byte{0x00, 0x2a})
	r.SetSwap(true)
	if got, err := r.Int16(); err != nil || got != 42 {
		t.Fatalf("Int16() with swap = %d, %v; want 42, nil", got, err)
	}
}

func TestReaderInt16Truncated(t *testing.T) {
	r := NewReader([]byte{0x2a})
	if _, err := r.Int16(); err == nil {
		t.Fatal("Int16() on 1-byte input: want error, got nil")
	}
}

func TestReaderUint64(t *testing.T) {
	// A DAWG edge record: letter=9, eow set (bit 9), next_node=1222,
	// with flag_start_bit=7 => raw 0x0000000000131a09.
	r := NewReader([]byte{0x09, 0x1a, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00})
	got, err := r.Uint64()
	if err != nil || got != 0x131a09 {
		t.Fatalf("Uint64() = %#x, %v; want 0x131a09, nil", got, err)
	}
}

func TestReaderUint64Swapped(t *testing.T) {
	r := NewReader([]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x13, 0x1a, 0x09})
	r.SetSwap(true)
	got, err := r.Uint64()
	if err != nil || got != 0x131a09 {
		t.Fatalf("Uint64() with swap = %#x, %v; want 0x131a09, nil", got, err)
	}
}

func TestReaderFloat32(t *testing.T) {
	// learning_rate_ in eng.traineddata is float32 0.001 == 0x3a83126f.
	r := NewReader([]byte{0x6f, 0x12, 0x83, 0x3a})
	got, err := r.Float32()
	if err != nil || got != float32(0.001) {
		t.Fatalf("Float32() = %v, %v; want 0.001, nil", got, err)
	}
}

func TestReaderFloat32Swapped(t *testing.T) {
	r := NewReader([]byte{0x3a, 0x83, 0x12, 0x6f})
	r.SetSwap(true)
	got, err := r.Float32()
	if err != nil || got != float32(0.001) {
		t.Fatalf("Float32() with swap = %v, %v; want 0.001, nil", got, err)
	}
}
