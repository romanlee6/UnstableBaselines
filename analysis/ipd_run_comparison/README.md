# Three-run IPD comparison

## Main findings

- **Learning speed:** first stable entry within 15% of the final outcome signature: Vanilla update 63, Broadcast update 190, Predict update 38.
- **Self-play equilibrium signature:** Vanilla ends near (+0.05, -0.05) and Predict near (+0.01, -0.01), both symmetric/tied. Predict is directly identified as mutual cooperation (98.5% CC). Vanilla cannot be split into CC versus DD from its logs.
- **Broadcast equilibrium:** ends near (-0.97, +0.97); role 1 almost always wins. Removing within-turn message ordering therefore did not remove the learned role asymmetry in this run.
- **Unseen-agent evaluation:** final terminal reward is Vanilla -0.66, Broadcast -0.97, Predict -0.03. Predict is close to neutral while both baselines lose, but its opponent differs, so this is suggestive rather than a controlled ranking.
- **Prediction mechanism:** final prediction accuracy is 98.9% for role 0 and 98.7% for role 1, alongside 98.5% mutual cooperation. Phase-specific rewards make typed communication, opponent modeling, and action choice separate learning problems.

## Interpretation caveats

- Vanilla and Broadcast do not log actions, so symmetric terminal outcomes cannot distinguish CC from DD.
- Unseen opponents differ: Vanilla/Broadcast use Gemini 3.1 Flash Lite; Predict uses DeepSeek-V4-flash.
- All curves are single-run descriptive results; no seed-level uncertainty intervals are available.
- Logged collection metrics are rolling means over up to 512 samples, so transitions are intentionally lagged.
- The Predict cooperate/defect scalar keys are malformed by a tracker substring check and are excluded; mutual_cooperation comes from environment step_info.

## Figures

1. `figure_1_learning_speed`: outcome curves and normalized distance to final attractor.
2. `figure_2_equilibrium_type`: role-pair terminal outcome trajectories and endpoints.
3. `figure_3_unseen_agent`: fixed-unseen-opponent evaluation curves and endpoint bars.
4. `figure_4_predict_phases`: phase validity, prediction/coordination, and phase reward channels.
