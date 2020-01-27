// flat surfaces
#define ISd_3d_0_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsj_ - nsk_] == 1 || cont_[index - nsj_] == 1 ||                    \
     cont_[index - nsj_ + sk_] == 1 || cont_[index_ - nsk_] == 1 ||                    \
     cont_[index_] == 1 || cont_[index_ + sk_] == 1 ||                                 \
     cont_[index_ + sj_ - nsk_] == 1 || cont_[index_ + sj_] == 1 ||                    \
     cont_[index_ + sj_ + sk_] == 1))
#define ISd_3d_1_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsk_] == 1 || cont_[index - nsi_] == 1 ||                    \
     cont_[index - nsi_ + sk_] == 1 || cont_[index_ - nsk_] == 1 ||                    \
     cont_[index_] == 1 || cont_[index_ + sk_] == 1 ||                                 \
     cont_[index + si_ - nsk_] == 1 || cont_[index + si_] == 1 ||                      \
     cont_[index + si_ + sk_] == 1))
#define ISd_3d_2_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsj_] == 1 || cont_[index - nsi_] == 1 ||                    \
     cont_[index - nsi_ + sj_] == 1 || cont_[index - nsj_] == 1 ||                     \
     cont_[index_] == 1 || cont_[index_ + sj_] == 1 ||                                 \
     cont_[index + si_ - nsj_] == 1 || cont_[index + si_] == 1 ||                      \
     cont_[index + si_ + sj_] == 1))

// diagonals loop i
#define ISd_3d_3_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsj_ - nsk_] == 1 || cont_[index - nsi_] == 1 ||             \
     cont_[index - nsi_ + sj_ + sk_] == 1 || cont_[index - nsj_ - nsk_] == 1 ||        \
     cont_[index_] == 1 || cont_[index_ + sj_ + sk_] == 1 ||                           \
     cont_[index + si_ - nsj_ - nsk_] == 1 || cont_[index + si_] == 1 ||               \
     cont_[index + si_ + sj_ + sk_] == 1))

#define ISd_3d_4_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsj_ + sk_] == 1 || cont_[index - nsi_] == 1 ||              \
     cont_[index - nsi_ + sj_ - nsk_] == 1 || cont_[index - nsj_ + sk_] == 1 ||        \
     cont_[index_] == 1 || cont_[index_ + sj_ - nsk_] == 1 ||                          \
     cont_[index + si_ - nsj_ + sk_] == 1 || cont_[index + si_] == 1 ||                \
     cont_[index + si_ + sj_ - nsk_] == 1))

// diagonals loop j
#define ISd_3d_5_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsj_ - nsk_] == 1 || cont_[index - nsj_] == 1 ||             \
     cont_[index + si_ - nsj_ + sk_] == 1 || cont_[index - nsi_ - nsk_] == 1 ||        \
     cont_[index_] == 1 || cont_[index + si_ + sk_] == 1 ||                            \
     cont_[index - nsi_ + sj_ - nsk_] == 1 || cont_[index_ + sj_] == 1 ||              \
     cont_[index + si_ + sj_ + sk_] == 1))

#define ISd_3d_6_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsj_ + sk_] == 1 || cont_[index - nsj_] == 1 ||              \
     cont_[index + si_ - nsj_ - nsk_] == 1 || cont_[index - nsi_ + sk_] == 1 ||        \
     cont_[index_] == 1 || cont_[index + si_ - nsk_] == 1 ||                           \
     cont_[index - nsi_ + sj_ + sk_] == 1 || cont_[index_ + sj_] == 1 ||               \
     cont_[index + si_ + sj_ - nsk_] == 1))

// diagonals loop k
#define ISd_3d_7_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ - nsj_ - nsk_] == 1 || cont_[index_ - nsk_] == 1 ||            \
     cont_[index + si_ + sj_ - nsk_] == 1 || cont_[index - nsi_ - nsj_] == 1 ||        \
     cont_[index_] == 1 || cont_[index + si_ + sj_] == 1 ||                            \
     cont_[index - nsi_ - nsj_ + sk_] == 1 || cont_[index_ + sk_] == 1 ||              \
     cont_[index + si_ + sj_ + sk_] == 1))

#define ISd_3d_8_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)                 \
  (!(cont_[index - nsi_ + sj_ - nsk_] == 1 || cont_[index_ - nsk_] == 1 ||             \
     cont_[index + si_ - nsj_ - nsk_] == 1 || cont_[index - nsi_ + sj_] == 1 ||        \
     cont_[index_] == 1 || cont_[index + si_ - nsj_] == 1 ||                           \
     cont_[index - nsi_ + sj_ + sk_] == 1 || cont_[index_ + sk_] == 1 ||               \
     cont_[index + si_ - nsj_ + sk_] == 1))

#define ISd_3d_any_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)               \
  (!(ISd_3d_0_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_1_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_2_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_3_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_4_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_5_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_6_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_7_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_) ||                 \
     ISd_3d_8_bidir(cont_, index_, nsi_, si_, nsj_, sj_, nsk_, sk_)))
