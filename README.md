# Transfer Oracle - Football Player Replacement Recommendation System

## Overview
Transfer Oracle is an intelligent recommendation system designed to identify football player replacements based on playing style archetypes using unsupervised machine learning. It leverages player statistics,  and clustering techniques to discover and classify player archetypes and then recommends the most similar players as replacements.

## Features
- Uses K-Means clustering to discover player archetypes within each position.
- Employs K-Nearest Neighbors (KNN) to find similar players filtered by position and archetype.
- Incorporates market values to recommend affordable replacements within budget constraints.
- Pre-computes archetypes for all players for efficient runtime search.
- Enables evaluation with silhouette scores, cluster characteristics, and domain validation.

## Motivation
The football transfer market is complex and highly competitive. Clubs need data-driven tools to identify exact player replacements who fit the tactical style and budget. Transfer Oracle addresses this need by combining detailed player stats analysis with machine learning clustering to recommend the most suitable candidates.

## Data Sources
- Player performance and stats from FBref and Kaggle dataset:
    https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025

## Methodology
1. **Offline Setup (Once):**
   - Load player data.
   - Run K-Means clustering by position to assign archetypes.
   - Save enriched player database with archetypes.

2. **Online Runtime:**
   - Lookup input player's archetype from the database.
   - Filter candidates by position and archetype.
   - Use KNN to find the closest player matches.
   - Apply post hoc constraints to recommend affordable replacements.
