\
# Rank-Tail Residual Risk DFL

Rank-Tail은 CAPM residual의 평균적 선형 covariance 대신
**동시 downside-tail dependence**를 residual risk로 정의한다.

## Risk definition

1. Within-asset residual rank:
   `u = rank(epsilon) / (T + 1)`

2. Continuous lower-tail score:
   `z = max(0, (q - u) / q)`

3. Time-centered tail score covariance:
   `G = Cov_t(z)`

4. Rank-Tail dependence:
   `R_RT = D_G^{-1/2} G D_G^{-1/2}`

5. Residual variance scaling:
   `A_RT = D_sigma R_RT D_sigma`

6. Effective covariance:
   `Sigma_eff = Sigma_market + A_RT + jitter I`

The diagonal of `A_RT` equals the ordinary residual variance, while the
off-diagonal structure is replaced by downside-tail rank dependence.

## Initial experiment

`q in {0.10, 0.20, 0.30}`, seeds `{42,43,44}`.
