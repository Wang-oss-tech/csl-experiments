def compute_iter(Mt, Kt, Nt):
      d = bin((3 * Kt >> 1) ^ (Kt >> 1)).count('1')
      b_load = 2 * d - 1 - (Kt % 2) + 8
      fmacs = (Mt+1) * 2  # 2 cycles per element (execute + idle on avg)
      dsd_increment = 14 + 3  # 12 ST16 + 2 ADD16 + 3 NOP
      loop_control = 3  # SUB16 + MOV16 + JNC
      return Kt * Nt * (b_load + fmacs + dsd_increment + loop_control)




