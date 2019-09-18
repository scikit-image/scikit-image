// flat surfaces
#define SId_3d_0(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index_ - sj_ - sk_] == 1 && cont_[index_ - sj_] == 1 &&                       \
   cont_[index_ - sj_ + sk_] == 1 && cont_[index_ - sk_] == 1 && cont_[index_] == 1 && \
   cont_[index_ + sk_] == 1 && cont_[index_ + sj_ - sk_] == 1 &&                       \
   cont_[index_ + sj_] == 1 && cont_[index_ + sj_ + sk_] == 1)

#define SId_3d_1(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sk_] == 1 && cont_[index - si_] == 1 &&                         \
   cont_[index - si_ + sk_] == 1 && cont_[index_ - sk_] == 1 && cont_[index_] == 1 &&  \
   cont_[index_ + sk_] == 1 && cont_[index + si_ - sk_] == 1 &&                        \
   cont_[index + si_] == 1 && cont_[index + si_ + sk_] == 1)

#define SId_3d_2(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sj_] == 1 && cont_[index - si_] == 1 &&                         \
   cont_[index - si_ + sj_] == 1 && cont_[index - sj_] == 1 && cont_[index_] == 1 &&   \
   cont_[index_ + sj_] == 1 && cont_[index + si_ - sj_] == 1 &&                        \
   cont_[index + si_] == 1 && cont_[index + si_ + sj_] == 1)

// diagonals loop i
#define SId_3d_3(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sj_ - sk_] == 1 && cont_[index - si_] == 1 &&                   \
   cont_[index - si_ + sj_ + sk_] == 1 && cont_[index - sj_ - sk_] == 1 &&             \
   cont_[index_] == 1 && cont_[index_ + sj_ + sk_] == 1 &&                             \
   cont_[index + si_ - sj_ - sk_] == 1 && cont_[index + si_] == 1 &&                   \
   cont_[index + si_ + sj_ + sk_] == 1)

#define SId_3d_4(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sj_ + sk_] == 1 && cont_[index - si_] == 1 &&                   \
   cont_[index - si_ + sj_ - sk_] == 1 && cont_[index - sj_ + sk_] == 1 &&             \
   cont_[index_] == 1 && cont_[index_ + sj_ - sk_] == 1 &&                             \
   cont_[index + si_ - sj_ + sk_] == 1 && cont_[index + si_] == 1 &&                   \
   cont_[index + si_ + sj_ - sk_] == 1)

// diagonals loop j_
#define SId_3d_5(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sj_ - sk_] == 1 && cont_[index - sj_] == 1 &&                   \
   cont_[index + si_ - sj_ + sk_] == 1 && cont_[index - si_ - sk_] == 1 &&             \
   cont_[index_] == 1 && cont_[index + si_ + sk_] == 1 &&                              \
   cont_[index - si_ + sj_ - sk_] == 1 && cont_[index_ + sj_] == 1 &&                  \
   cont_[index + si_ + sj_ + sk_] == 1)

#define SId_3d_6(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sj_ + sk_] == 1 && cont_[index - sj_] == 1 &&                   \
   cont_[index + si_ - sj_ - sk_] == 1 && cont_[index - si_ + sk_] == 1 &&             \
   cont_[index_] == 1 && cont_[index + si_ - sk_] == 1 &&                              \
   cont_[index - si_ + sj_ + sk_] == 1 && cont_[index_ + sj_] == 1 &&                  \
   cont_[index + si_ + sj_ - sk_] == 1)

// diagonals loop
#define SId_3d_7(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ - sj_ - sk_] == 1 && cont_[index_ - sk_] == 1 &&                  \
   cont_[index + si_ + sj_ - sk_] == 1 && cont_[index - si_ - sj_] == 1 &&             \
   cont_[index_] == 1 && cont_[index + si_ + sj_] == 1 &&                              \
   cont_[index - si_ - sj_ + sk_] == 1 && cont_[index_ + sk_] == 1 &&                  \
   cont_[index + si_ + sj_ + sk_] == 1)

#define SId_3d_8(cont_, index_, si_, sj_, sk_)                                         \
  (cont_[index - si_ + sj_ - sk_] == 1 && cont_[index_ - sk_] == 1 &&                  \
   cont_[index + si_ - sj_ - sk_] == 1 && cont_[index - si_ + sj_] == 1 &&             \
   cont_[index_] == 1 && cont_[index + si_ - sj_] == 1 &&                              \
   cont_[index - si_ + sj_ + sk_] == 1 && cont_[index_ + sk_] == 1 &&                  \
   cont_[index + si_ - sj_ + sk_] == 1)

#define SId_3d_any(cont_, index_, si_, sj_, sk_)                                       \
  (SId_3d_0(cont_, index_, si_, sj_, sk_) || SId_3d_1(cont_, index_, si_, sj_, sk_) || \
   SId_3d_2(cont_, index_, si_, sj_, sk_) || SId_3d_3(cont_, index_, si_, sj_, sk_) || \
   SId_3d_4(cont_, index_, si_, sj_, sk_) || SId_3d_5(cont_, index_, si_, sj_, sk_) || \
   SId_3d_6(cont_, index_, si_, sj_, sk_) || SId_3d_7(cont_, index_, si_, sj_, sk_) || \
   SId_3d_8(cont_, index_, si_, sj_, sk_))

// flat surfaces
#define ISd_3d_0(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - sj_ - sk_] == 1 || cont_[index - sj_] == 1 ||                       \
     cont_[index - sj_ + sk_] == 1 || cont_[index_ - sk_] == 1 ||                      \
     cont_[index_] == 1 || cont_[index_ + sk_] == 1 ||                                 \
     cont_[index_ + sj_ - sk_] == 1 || cont_[index_ + sj_] == 1 ||                     \
     cont_[index_ + sj_ + sk_] == 1))
#define ISd_3d_1(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sk_] == 1 || cont_[index - si_] == 1 ||                       \
     cont_[index - si_ + sk_] == 1 || cont_[index_ - sk_] == 1 ||                      \
     cont_[index_] == 1 || cont_[index_ + sk_] == 1 ||                                 \
     cont_[index + si_ - sk_] == 1 || cont_[index + si_] == 1 ||                       \
     cont_[index + si_ + sk_] == 1))
#define ISd_3d_2(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sj_] == 1 || cont_[index - si_] == 1 ||                       \
     cont_[index - si_ + sj_] == 1 || cont_[index - sj_] == 1 || cont_[index_] == 1 || \
     cont_[index_ + sj_] == 1 || cont_[index + si_ - sj_] == 1 ||                      \
     cont_[index + si_] == 1 || cont_[index + si_ + sj_] == 1))

// diagonals loop i
#define ISd_3d_3(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sj_ - sk_] == 1 || cont_[index - si_] == 1 ||                 \
     cont_[index - si_ + sj_ + sk_] == 1 || cont_[index - sj_ - sk_] == 1 ||           \
     cont_[index_] == 1 || cont_[index_ + sj_ + sk_] == 1 ||                           \
     cont_[index + si_ - sj_ - sk_] == 1 || cont_[index + si_] == 1 ||                 \
     cont_[index + si_ + sj_ + sk_] == 1))

#define ISd_3d_4(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sj_ + sk_] == 1 || cont_[index - si_] == 1 ||                 \
     cont_[index - si_ + sj_ - sk_] == 1 || cont_[index - sj_ + sk_] == 1 ||           \
     cont_[index_] == 1 || cont_[index_ + sj_ - sk_] == 1 ||                           \
     cont_[index + si_ - sj_ + sk_] == 1 || cont_[index + si_] == 1 ||                 \
     cont_[index + si_ + sj_ - sk_] == 1))

// diagonals loop j
#define ISd_3d_5(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sj_ - sk_] == 1 || cont_[index - sj_] == 1 ||                 \
     cont_[index + si_ - sj_ + sk_] == 1 || cont_[index - si_ - sk_] == 1 ||           \
     cont_[index_] == 1 || cont_[index + si_ + sk_] == 1 ||                            \
     cont_[index - si_ + sj_ - sk_] == 1 || cont_[index_ + sj_] == 1 ||                \
     cont_[index + si_ + sj_ + sk_] == 1))

#define ISd_3d_6(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sj_ + sk_] == 1 || cont_[index - sj_] == 1 ||                 \
     cont_[index + si_ - sj_ - sk_] == 1 || cont_[index - si_ + sk_] == 1 ||           \
     cont_[index_] == 1 || cont_[index + si_ - sk_] == 1 ||                            \
     cont_[index - si_ + sj_ + sk_] == 1 || cont_[index_ + sj_] == 1 ||                \
     cont_[index + si_ + sj_ - sk_] == 1))

// diagonals loop k
#define ISd_3d_7(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ - sj_ - sk_] == 1 || cont_[index_ - sk_] == 1 ||                \
     cont_[index + si_ + sj_ - sk_] == 1 || cont_[index - si_ - sj_] == 1 ||           \
     cont_[index_] == 1 || cont_[index + si_ + sj_] == 1 ||                            \
     cont_[index - si_ - sj_ + sk_] == 1 || cont_[index_ + sk_] == 1 ||                \
     cont_[index + si_ + sj_ + sk_] == 1))

#define ISd_3d_8(cont_, index_, si_, sj_, sk_)                                         \
  (!(cont_[index - si_ + sj_ - sk_] == 1 || cont_[index_ - sk_] == 1 ||                \
     cont_[index + si_ - sj_ - sk_] == 1 || cont_[index - si_ + sj_] == 1 ||           \
     cont_[index_] == 1 || cont_[index + si_ - sj_] == 1 ||                            \
     cont_[index - si_ + sj_ + sk_] == 1 || cont_[index_ + sk_] == 1 ||                \
     cont_[index + si_ - sj_ + sk_] == 1))

#define ISd_3d_any(cont_, index_, si_, sj_, sk_)                                       \
  (!(ISd_3d_0(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_1(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_2(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_3(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_4(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_5(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_6(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_7(cont_, index_, si_, sj_, sk_) ||                                         \
     ISd_3d_8(cont_, index_, si_, sj_, sk_)))
