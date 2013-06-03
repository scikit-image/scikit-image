/* Intrinsic declarations */
#if defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__MMX__)
#include <mmintrin.h>
#elif defined(__ALTIVEC__)
#include <altivec.h>
#endif

/* Compiler peculiarities */
#if defined(__GNUC__)
#include <stdint.h>
#elif defined(_MSC_VER)
#define inline __inline
typedef unsigned __int16 uint16_t;
#endif

/**
 * Add 16 unsigned 16-bit integers using SSE2, MMX or Altivec, if
 * available.
 */
#if defined(__SSE2__)
static inline void add16(uint16_t *dest, uint16_t *src)
{
    __m128i *d, *s;
    d = (__m128i *) dest;
    s = (__m128i *) src;
    *d = _mm_add_epi16(*d, *s);
    d++; s++;
    *d = _mm_add_epi16(*d, *s);
}
#elif defined(__MMX__)
static inline void add16(uint16_t *dest, uint16_t *src)
{
    __m64 *d, *s;
    d = (__m64 *) dest;
    s = (__m64 *) src;
    *d = _mm_add_pi16(*d, *s);
    d++; s++;
    *d = _mm_add_pi16(*d, *s);
    d++; s++;
    *d = _mm_add_pi16(*d, *s);
    d++; s++;
    *d = _mm_add_pi16(*d, *s);
}
#elif defined(__ALTIVEC__)
static inline void add16(uint16_t *dest, uint16_t *src)
{
    vector unsigned short *d, *s;
    d = (vector unsigned short *) dest;
    s = (vector unsigned short *) src;
    *d = vec_add(*d, *s);
    d++; s++;
    *d = vec_add(*d, *s);
}
#else
static inline void add16(uint16_t *dest, uint16_t *src)
{
    int i;

    for (i = 0; i < 16; i++) dest[i] += src[i];
}
#endif

/**
 * Subtract 16 unsigned 16-bit integers using SSE2, MMX or Altivec, if
 * available.
 */
#if defined(__SSE2__)
static inline void sub16(uint16_t *dest, uint16_t *src)
{
    __m128i *d, *s;
    d = (__m128i *) dest;
    s = (__m128i *) src;
    *d = _mm_sub_epi16(*d, *s);
    d++; s++;
    *d = _mm_sub_epi16(*d, *s);
}
#elif defined(__MMX__)
static inline void sub16(uint16_t *dest, uint16_t *src)
{
    __m64 *d, *s;
    d = (__m64 *) dest;
    s = (__m64 *) src;
    *d = _mm_sub_pi16(*d, *s);
    d++; s++;
    *d = _mm_sub_pi16(*d, *s);
    d++; s++;
    *d = _mm_sub_pi16(*d, *s);
    d++; s++;
    *d = _mm_sub_pi16(*d, *s);
}
#elif defined(__ALTIVEC__)
static inline void sub16(uint16_t *dest, uint16_t *src)
{
    vector unsigned short *d, *s;
    d = (vector unsigned short *) dest;
    s = (vector unsigned short *) src;
    *d = vec_sub(*d, *s);
    d++; s++;
    *d = vec_sub(*d, *s);
}
#else
static inline void sub16(uint16_t *dest, uint16_t *src)
{
    int i;

    for (i = 0; i < 16; i++) dest[i] -= src[i];
}
#endif
