// Regenerates Leptonica goldens. Manual step; see ../../../Makefile `goldens`.
// Each golden is a raw dump: width, height, depth as int32 LE, then packed rows.
#include <stdio.h>
#include <stdlib.h>
#include <leptonica/allheaders.h>

static void dump(const char *path, PIX *p) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int32_t hdr[3] = { pixGetWidth(p), pixGetHeight(p), pixGetDepth(p) };
    fwrite(hdr, sizeof(int32_t), 3, f);
    l_uint32 *data = pixGetData(p);
    int wpl = pixGetWpl(p);
    fwrite(data, sizeof(l_uint32), (size_t)wpl * pixGetHeight(p), f);
    fclose(f);
    printf("wrote %s (%dx%d d=%d)\n", path, hdr[0], hdr[1], hdr[2]);
}

// dumpOwned writes p as <outdir>/<name> and destroys it.
static void dumpOwned(const char *outdir, const char *name, PIX *p) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", outdir, name);
    dump(path, p);
    pixDestroy(&p);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: gen <input.png> <outdir>\n"); return 2; }
    PIX *src = pixRead(argv[1]);
    if (!src) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    PIX *gray = pixConvertTo8(src, 0);

    char path[512];
    snprintf(path, sizeof path, "%s/gray.bin", argv[2]);
    dump(path, gray);

    // Otsu, whole-image (no tiling), no smoothing.
    PIX *otsu = NULL;
    pixOtsuAdaptiveThreshold(gray, pixGetWidth(gray), pixGetHeight(gray), 0, 0, 0.0, NULL, &otsu);
    snprintf(path, sizeof path, "%s/otsu.bin", argv[2]);
    dump(path, otsu);

    // Sauvola over a 17x17 window (whsize 8), addborder=1 so Leptonica mirrors
    // the border itself. k is 11/32, which is exact in both float32 (the type
    // of Leptonica's factor) and float64 (the type of the Go k), so the two
    // implementations cannot disagree about the constant.
    PIX *sauvola = NULL;
    pixSauvolaBinarize(gray, 8, 0.34375f, 1, NULL, NULL, NULL, &sauvola);
    snprintf(path, sizeof path, "%s/sauvola.bin", argv[2]);
    dump(path, sauvola);

    // Rasterop. Two 320x400 1bpp operands cut from the binarizations above, so
    // the goldens are self-contained: the Go test loads the operands as well as
    // the results and never re-derives them.
    //
    // The destination rect (-40,-30,400,500) with source origin (60,80) is
    // deliberately impossible to satisfy: it starts left of and above the
    // destination and overhangs both images to the right and below. Every one of
    // pixRasterop's four clipping adjustments fires, which is where hand-rolled
    // raster ops diverge.
    BOX *crop = boxCreate(300, 150, 320, 400);
    PIX *ropDst = pixClipRectangle(otsu, crop, NULL);
    PIX *ropSrc = pixClipRectangle(sauvola, crop, NULL);
    snprintf(path, sizeof path, "%s/rop_dst_in.bin", argv[2]);
    dump(path, ropDst);
    snprintf(path, sizeof path, "%s/rop_src_in.bin", argv[2]);
    dump(path, ropSrc);

    static const struct { const char *name; l_int32 op; } ops[] = {
        { "rop_set.bin",    PIX_SET },
        { "rop_clr.bin",    PIX_CLR },
        { "rop_copy.bin",   PIX_SRC },
        { "rop_notsrc.bin", PIX_NOT(PIX_SRC) },
        { "rop_or.bin",     PIX_SRC | PIX_DST },
        { "rop_and.bin",    PIX_SRC & PIX_DST },
        { "rop_xor.bin",    PIX_SRC ^ PIX_DST },
    };
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        PIX *d = pixCopy(NULL, ropDst);
        pixRasterop(d, -40, -30, 400, 500, ops[i].op, ropSrc, 60, 80);
        dumpOwned(argv[2], ops[i].name, d);
    }

    // The other clipping direction: a negative source origin, which shifts the
    // destination rect right and down instead of the source.
    PIX *negsrc = pixCopy(NULL, ropDst);
    pixRasterop(negsrc, 200, 250, 200, 300, PIX_SRC, ropSrc, -60, -80);
    dumpOwned(argv[2], "rop_negsrc.bin", negsrc);

    // Brick morphology over the whole Otsu image, with Leptonica's default
    // asymmetric boundary condition (everything outside the image is OFF).
    // 4x2 is included because an even-sided brick puts its origin off centre
    // (at hsize/2, vsize/2), making dilation and erosion mirror images rather
    // than the same window.
    dumpOwned(argv[2], "dilate_5x3.bin", pixDilateBrick(NULL, otsu, 5, 3));
    dumpOwned(argv[2], "dilate_4x2.bin", pixDilateBrick(NULL, otsu, 4, 2));
    dumpOwned(argv[2], "erode_3x7.bin", pixErodeBrick(NULL, otsu, 3, 7));
    dumpOwned(argv[2], "open_5x5.bin", pixOpenBrick(NULL, otsu, 5, 5));
    dumpOwned(argv[2], "close_7x3.bin", pixCloseBrick(NULL, otsu, 7, 3));

    boxDestroy(&crop);
    pixDestroy(&ropDst); pixDestroy(&ropSrc);
    pixDestroy(&src); pixDestroy(&gray); pixDestroy(&otsu); pixDestroy(&sauvola);
    return 0;
}
