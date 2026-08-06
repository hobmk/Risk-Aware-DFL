# Residual Collective Risk-Aware DFL

이 구현은 rolling CAPM 잔차에서 residual correlation network를 계산하고, 이를 covariance 단위로 변환한 뒤 MVO의 위험행렬에 추가한다.

## 핵심 정의

- `C_res,t = Corr(epsilon_t)`
- `C_bar,t = (1-rho) C_res,t + rho I`
- `A_res,t = tr(Sigma_t)/N * C_bar,t`
- `Sigma_eff,t = Sigma_t + eta A_res,t`

`correlation_scaling="trace"`이면 `tr(A_res,t) = tr(Sigma_t)`이므로 eta가 두 위험행렬의 상대적 비중으로 해석된다.

## 폴더 구조

```text
rcr_dfl/
├── src/
│   ├── capm.py
│   ├── dataset.py
│   ├── residual_risk.py
│   ├── effective_covariance.py
│   ├── model.py
│   ├── optimization.py
│   ├── decision_model.py
│   ├── losses.py
│   └── trainer.py
├── scripts/
│   ├── check_core.py
│   ├── check_dataset.py
│   ├── analyze_risk_matrices.py
│   ├── check_rcr_pipeline.py
│   └── train_rcr_combined.py
└── tests/
```

## 실행 순서

저장소 루트에서 실행한다.

```powershell
python -m implementations.rcr_dfl.scripts.check_core
```

```powershell
python -m implementations.rcr_dfl.scripts.check_dataset `
    --price-csv data/raw/dow30_adjusted_close.csv `
    --lookback 60 `
    --eta 0.5
```

```powershell
python -m implementations.rcr_dfl.scripts.analyze_risk_matrices `
    --price-csv data/raw/dow30_adjusted_close.csv `
    --lookback 60 `
    --eta 0.5 `
    --output-dir implementations/rcr_dfl/outputs/risk_matrix_analysis
```

```powershell
python -m implementations.rcr_dfl.scripts.check_rcr_pipeline --eta 0.5
```

```powershell
python -m implementations.rcr_dfl.scripts.train_rcr_combined `
    --price-csv data/raw/dow30_adjusted_close.csv `
    --eta 0.5 `
    --epochs 3 `
    --max-train-batches 5 `
    --max-validation-batches 2 `
    --max-test-batches 2 `
    --output-dir implementations/rcr_dfl/outputs/smoke_eta_050 `
    --overwrite
```

## 테스트

```powershell
python -m pytest implementations/rcr_dfl/tests -q
```

`decision_model`과 `optimization` 테스트에는 `cvxpy`, `cvxpylayers`가 필요하다.
