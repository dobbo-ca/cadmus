LEPT_PREFIX := $(shell brew --prefix leptonica)
CFLAGS := -I$(LEPT_PREFIX)/include -O2
LDFLAGS := -L$(LEPT_PREFIX)/lib -lleptonica

all: gen
	./gen ../input/scan.png ..

gen: gen.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

scaleline: scaleline.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f gen scaleline
