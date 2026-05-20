# e-He LXCat Elastic Momentum-Transfer Audit

This folder contains a local copy of the LXCat download for electron-helium elastic
momentum-transfer cross sections and a reproducible audit summary.

Use this data for the current `sigma_ep` closure. Do not use the Cs+ / He ion-neutral
Q(01) data for electron momentum transport.

Representative median values across the included LXCat databases:

| T_e (K) | E = kBT (eV) | median sigma (m^2) | database spread |
| ---: | ---: | ---: | ---: |
| 2250 | 0.19389 | 6.154770e-20 | 1.33% |
| 3000 | 0.25852 | 6.280575e-20 | 1.57% |
| 4300 | 0.370545 | 6.450369e-20 | 1.85% |
| 4900 | 0.422249 | 6.510912e-20 | 1.95% |
| 6200 | 0.534275 | 6.611532e-20 | 2.09% |

At 4300 K, the median LXCat value is close to the classic e-He momentum-transfer
range and about 16x larger than the current legacy `sigma_ep` value.
