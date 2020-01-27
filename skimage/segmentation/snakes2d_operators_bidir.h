// diagonal
#define ISd_2d_0_bidir(cont_, idx_, nsi_, si_, nsj_, sj_)                              \
  (!(cont_[idx_ - nsi_ - nsj_] == 1 || cont_[idx_] == 1 ||                             \
     cont_[idx_ + si_ + sj_] == 1))

// const y
#define ISd_2d_1_bidir(cont_, idx_, nsi_, si_, nsj_, sj_)                              \
  (!(cont_[idx_ - nsi_] == 1 || cont_[idx_] == 1 || cont_[idx_ + si_] == 1))

// diagonal
#define ISd_2d_2_bidir(cont_, idx_, nsi_, si_, nsj_, sj_)                              \
  (!(cont_[idx_ - nsi_ + sj_] == 1 || cont_[idx_] == 1 ||                              \
     cont_[idx_ + si_ - nsj_] == 1))

// const x
#define ISd_2d_3_bidir(cont_, idx_, nsi_, si_, nsj_, sj_)                              \
  (!(cont_[idx_ - nsj_] == 1 || cont_[idx_] == 1 || cont_[idx_ + sj_] == 1))

#define ISd_2d_any_bidir(cont_, idx_, nsi_, si_, nsj_, sj_)                            \
  (!(ISd_2d_0_bidir(cont_, idx_, nsi_, si_, nsj_, sj_) ||                              \
     ISd_2d_1_bidir(cont_, idx_, nsi_, si_, nsj_, sj_) ||                              \
     ISd_2d_2_bidir(cont_, idx_, nsi_, si_, nsj_, sj_) ||                              \
     ISd_2d_3_bidir(cont_, idx_, nsi_, si_, nsj_, sj_)))
