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
