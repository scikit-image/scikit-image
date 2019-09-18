// diagonal
#define SId_2d_0(cont_, idx_, si_, sj_)                                                \
  (cont_[idx_ - si_ - sj_] == 1 && cont_[idx_] == 1 && cont_[idx_ + si_ + sj_] == 1)
#define ISd_2d_0(cont_, idx_, si_, sj_)                                                \
  (!(cont_[idx_ - si_ - sj_] == 1 || cont_[idx_] == 1 || cont_[idx_ + si_ + sj_] == 1))

// const y
#define SId_2d_1(cont_, idx_, si_, sj_)                                                \
  (cont_[idx_ - si_] == 1 && cont_[idx_] == 1 && cont_[idx_ + si_] == 1)
#define ISd_2d_1(cont_, idx_, si_, sj_)                                                \
  (!(cont_[idx_ - si_] == 1 || cont_[idx_] == 1 || cont_[idx_ + si_] == 1))

// diagonal
#define SId_2d_2(cont_, idx_, si_, sj_)                                                \
  (cont_[idx_ - si_ + sj_] == 1 && cont_[idx_] == 1 && cont_[idx_ + si_ - sj_] == 1)
#define ISd_2d_2(cont_, idx_, si_, sj_)                                                \
  (!(cont_[idx_ - si_ + sj_] == 1 || cont_[idx_] == 1 || cont_[idx_ + si_ - sj_] == 1))

// const x
#define SId_2d_3(cont_, idx_, si_, sj_)                                                \
  (cont_[idx_ - sj_] == 1 && cont_[idx_] == 1 && cont_[idx_ + sj_] == 1)
#define ISd_2d_3(cont_, idx_, si_, sj_)                                                \
  (!(cont_[idx_ - sj_] == 1 || cont_[idx_] == 1 || cont_[idx_ + sj_] == 1))

#define SId_2d_any(cont_, idx_, si_, sj_)                                              \
  (SId_2d_0(cont_, idx_, si_, sj_) || SId_2d_1(cont_, idx_, si_, sj_) ||               \
   SId_2d_2(cont_, idx_, si_, sj_) || SId_2d_3(cont_, idx_, si_, sj_))

#define ISd_2d_any(cont_, idx_, si_, sj_)                                              \
  (!(ISd_2d_0(cont_, idx_, si_, sj_) || ISd_2d_1(cont_, idx_, si_, sj_) ||             \
     ISd_2d_2(cont_, idx_, si_, sj_) || ISd_2d_3(cont_, idx_, si_, sj_)))
