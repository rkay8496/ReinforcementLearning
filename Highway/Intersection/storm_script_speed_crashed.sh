#!/bin/zsh

storm --prism ppo_roundabout_v0_10000.0_0.001.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_10000.0_0.0002.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_10000.0_0.0004.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_10000.0_0.0006.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_10000.0_0.0008.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"

storm --prism ppo_roundabout_v0_15000.0_0.001.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_15000.0_0.0002.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_15000.0_0.0004.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_15000.0_0.0006.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_15000.0_0.0008.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"

storm --prism ppo_roundabout_v0_20000.0_0.001.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_20000.0_0.0002.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_20000.0_0.0004.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_20000.0_0.0006.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_20000.0_0.0008.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"

storm --prism ppo_roundabout_v0_25000.0_0.001.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_25000.0_0.0002.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_25000.0_0.0004.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_25000.0_0.0006.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_25000.0_0.0008.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"

storm --prism ppo_roundabout_v0_30000.0_0.001.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_30000.0_0.0002.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_30000.0_0.0004.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_30000.0_0.0006.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"
storm --prism ppo_roundabout_v0_30000.0_0.0008.prism --prop "filter(avg, R{\"speed\"}=?[F \"crashed\"], \"safe\")"