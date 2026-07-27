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

// dumpConnComp writes a connected-component golden. Unlike the PIX goldens this
// is a table, not an image: int32 LE component count, then one record of five
// int32 LE per component (x, y, w, h, foreground pixel count), in the order
// pixConnComp emits them, which is raster order of each component's first
// foreground pixel.
static void dumpConnComp(const char *outdir, const char *name, PIX *p, l_int32 conn) {
    PIXA *pixa = NULL;
    BOXA *boxa = pixConnComp(p, &pixa, conn);
    if (!boxa) { fprintf(stderr, "pixConnComp(conn=%d) failed\n", conn); exit(1); }
    NUMA *counts = pixaCountPixels(pixa);
    if (!counts) { fprintf(stderr, "pixaCountPixels(conn=%d) failed\n", conn); exit(1); }

    char path[512];
    snprintf(path, sizeof path, "%s/%s", outdir, name);
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    int32_t n = boxaGetCount(boxa);
    fwrite(&n, sizeof n, 1, f);
    for (int32_t i = 0; i < n; i++) {
        l_int32 x, y, w, h, c;
        boxaGetBoxGeometry(boxa, i, &x, &y, &w, &h);
        numaGetIValue(counts, i, &c);
        int32_t rec[5] = { x, y, w, h, c };
        fwrite(rec, sizeof(int32_t), 5, f);
    }
    fclose(f);
    printf("wrote %s (%d components, conn=%d)\n", path, n, conn);

    numaDestroy(&counts);
    pixaDestroy(&pixa);
    boxaDestroy(&boxa);
}

// makeConnCompDiag builds a small operand whose component set differs between
// 4- and 8-connectivity: a diagonal staircase, two blocks touching only at a
// corner, a lone pixel, and a run along the top border. The scan image has no
// diagonal contacts anywhere, so its two connectivity goldens come out
// byte-identical and cannot catch an implementation that ignores the
// parameter. This one also pins the pathological cases the oracle should cover:
// a single-pixel component, and a component running to the image edge.
static PIX *makeConnCompDiag(void) {
    PIX *p = pixCreate(40, 40, 1);
    if (!p) { fprintf(stderr, "pixCreate failed\n"); exit(1); }
    for (int x = 0; x < 4; x++) pixSetPixel(p, x, 0, 1);
    pixSetPixel(p, 38, 1, 1);
    for (int i = 0; i < 16; i++) pixSetPixel(p, 2 + i, 2 + i, 1);
    for (int y = 20; y < 25; y++)
        for (int x = 20; x < 25; x++) pixSetPixel(p, x, y, 1);
    for (int y = 25; y < 30; y++)
        for (int x = 25; x < 30; x++) pixSetPixel(p, x, y, 1);
    return p;
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

    // Connected components over the whole Otsu image, both connectivities. The
    // input carries isolated 1-2px specks and a full-height vertical rule, so
    // the goldens cover single-pixel components and components running to the
    // image border as well as the ordinary text-line blobs.
    dumpConnComp(argv[2], "conncomp4.bin", otsu, 4);
    dumpConnComp(argv[2], "conncomp8.bin", otsu, 8);

    PIX *diag = makeConnCompDiag();
    dumpConnComp(argv[2], "conncomp_diag4.bin", diag, 4);
    dumpConnComp(argv[2], "conncomp_diag8.bin", diag, 8);
    dumpOwned(argv[2], "conncomp_diag_in.bin", diag);

    boxDestroy(&crop);
    pixDestroy(&ropDst); pixDestroy(&ropSrc);
    pixDestroy(&src); pixDestroy(&gray); pixDestroy(&otsu); pixDestroy(&sauvola);
    return 0;
}
