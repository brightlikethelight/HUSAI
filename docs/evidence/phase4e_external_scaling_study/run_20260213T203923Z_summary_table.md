# External Scaling Condition Table (Run 20260213T203923Z)

- CE-Bench matched-200 public baseline interpretability max: `47.951611585617066`

| condition | token_budget | hook_layer | d_sae | train EV | train L0 | SAEBench best-LLM AUC delta | CE-Bench interpretability max | CE-Bench interp delta vs matched-200 baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tok10000_layer0_dsae1024_seed42 | 10000 | 0 | 1024 | 0.9982019997501268 | 32.0 | -0.04903780812971936 | 6.1836693835258485 | -41.76794220209122 |
| tok10000_layer0_dsae2048_seed42 | 10000 | 0 | 2048 | 0.998099990811878 | 32.0 | -0.07341774510319299 | 7.5051356482505795 | -40.446475937366486 |
| tok10000_layer1_dsae1024_seed42 | 10000 | 1 | 1024 | 0.9973477099700362 | 32.0 | -0.09526819738578596 | 7.965026619434357 | -39.98658496618271 |
| tok10000_layer1_dsae2048_seed42 | 10000 | 1 | 2048 | 0.997303435911961 | 32.0 | -0.09643668125361105 | 10.38789572238922 | -37.56371586322784 |
| tok30000_layer0_dsae1024_seed42 | 30000 | 0 | 1024 | 0.9988613782775199 | 32.0 | -0.08369549712378399 | 6.440171301364899 | -41.51144028425217 |
| tok30000_layer0_dsae2048_seed42 | 30000 | 0 | 2048 | 0.9989387148129208 | 32.0 | -0.06875213594940133 | 7.378561475276947 | -40.57305011034012 |
| tok30000_layer1_dsae1024_seed42 | 30000 | 1 | 1024 | 0.998272660888211 | 32.0 | -0.0918547310182507 | 8.10938639640808 | -39.84222518920899 |
| tok30000_layer1_dsae2048_seed42 | 30000 | 1 | 2048 | 0.9983954800719961 | 32.0 | -0.09558630302151461 | 10.552940266132355 | -37.39867131948471 |

## Axis Aggregates

### By Token Budget
- `10000`: SAEBench delta mean `-0.07854010796807734`; CE-Bench interpretability mean `8.010431843400003`
- `30000`: SAEBench delta mean `-0.08497216677823766`; CE-Bench interpretability mean `8.12026485979557`

### By Hook Layer
- `0`: SAEBench delta mean `-0.06872579657652442`; CE-Bench interpretability mean `6.8768844521045684`
- `1`: SAEBench delta mean `-0.09478647816979058`; CE-Bench interpretability mean `9.253812251091004`

### By d_sae
- `1024`: SAEBench delta mean `-0.079964058414385`; CE-Bench interpretability mean `7.174563425183297`
- `2048`: SAEBench delta mean `-0.08354821633192999`; CE-Bench interpretability mean `8.956133278012276`
