import React, { useState } from "react";
import "./App.css";

import FW_RECS from "./data/FW_recs.json";
import MF_RECS from "./data/MF_recs.json";
import DF_RECS from "./data/DF_recs.json";
import GK_RECS from "./data/GK_recs.json";

const RECS_BY_POS = {
  FW: FW_RECS,
  MF: MF_RECS,
  DF: DF_RECS,
  GK: GK_RECS,
};

const PLAYER_LIST_BY_POS = {
  FW: Object.keys(FW_RECS || {}),
  MF: Object.keys(MF_RECS || {}),
  DF: Object.keys(DF_RECS || {}),
  GK: Object.keys(GK_RECS || {}),
};

const POSITION_ZONES = [
  { code: "FW", label: "Forwards", short: "FW" },
  { code: "MF", label: "Midfielders", short: "MF" },
  { code: "DF", label: "Defenders", short: "DF" },
  { code: "GK", label: "Goalkeepers", short: "GK" },
];

// Top-6 features per position (for labels + ordering on radar)
const RADAR_FEATURES_BY_POS = {
  FW: ["G+A-PK", "xG+xAG", "Sh/90", "Att 3rd_stats_possession", "G/Sh", "SoT/90"],
  MF: ["PrgP", "PrgC", "xA", "Mid 3rd_stats_possession", "Tkl+Int", "G+A-PK"],
  DF: ["Tkl+Int", "Clr", "Def 3rd_stats_possession", "PrgP", "PrgDist_stats_possession", "Cmp%"],
  GK: ["Save%", "/90", "CS%", "Stp%", "#OPA/90", "Cmp%_stats_keeper_adv"],
};

// Human-readable descriptions for each radar feature, per position
const RADAR_FEATURE_LEGENDS = {
  FW: {
    "G+A-PK": "Non-Penalty Goals + Assists/90min",
    "xG+xAG": "Expected Goals + Expected Assisted Goals/90min",
    "Sh/90": "Shots Total/90min",
    "Att 3rd_stats_possession": "Touches in attacking 1/3",
    "G/Sh": "Goals/Shot",
    "SoT/90": "Shots on Target/90min",
  },
  MF: {
    PrgP: "Progressive Passes",
    PrgC: "Progressive Carries",
    xA: "Expected Assists",
    "Mid 3rd_stats_possession": "Touches in middle 1/3",
    "Tkl+Int": "Tackles + Interceptions",
    "G+A-PK": "Non-Penalty Goals + Assists/90min",
  },
  DF: {
    "Tkl+Int": "Tackles + Interceptions",
    Clr: "Clearances",
    "Def 3rd_stats_possession": "Touches in defensive 1/3",
    PrgP: "Progressive Passes",
    "PrgDist_stats_possession": "Progressive Carrying Distance",
    "Cmp%": "Pass Completion %",
  },
  GK: {
    "Save%": "Save %",
    "/90": "Post-Shot xG minus Goals Allowed per 90",
    "CS%": "Clean Sheet Percentage",
    "Stp%": "Crosses Stopped %",
    "#OPA/90": "Defensive Actions Outside Penalty Area/90min",
    "Cmp%_stats_keeper_adv": "Passes Completed (Launched +40 yards)",
  },
};

// Angles for the 6 axes (degrees) – must match the radar SVG
const RADAR_ANGLES_DEG = [0, 60, 120, 180, 240, 300];

// Colors for base + up to 3 overlay series
const RADAR_SERIES_COLORS = [
  { stroke: "#3b82f6", fill: "rgba(59,130,246,0.24)" }, // base (query)
  { stroke: "#22c55e", fill: "rgba(34,197,94,0.20)" },  // overlay 1
  { stroke: "#eab308", fill: "rgba(234,179,8,0.20)" },  // overlay 2
  { stroke: "#ec4899", fill: "rgba(236,72,153,0.20)" }, // overlay 3
];

const RadarCard = ({
  player,
  pos,
  labels,
  baseValues,
  seriesLegend,
  overlaySeries,
  featureLegend,
}) => {
  const centerX = 110;
  const centerY = 110;
  const minR = 10;
  const maxR = 90;

  const hasBase =
    Array.isArray(baseValues) && baseValues.some((v) => typeof v === "number");

  // Take percentile-normalized value in [0,1] and boost it slightly
  // so high-end players stand out more on the radar.
  const normalizeForRadius = (v) => {
    if (typeof v !== "number") return 0;
    const clamped = Math.min(1, Math.max(0, v));
    const boosted = Math.sqrt(clamped); // accentuate high values
    return boosted;
  };

  const computePoints = (values) => {
    if (!values || !values.length) {
      // fallback polygon
      return [
        "110,35",
        "170,90",
        "155,155",
        "110,190",
        "60,165",
        "45,95",
      ].join(" ");
    }

    return RADAR_ANGLES_DEG.map((angleDeg, i) => {
      const scaled = normalizeForRadius(values[i]);
      const radius = minR + scaled * (maxR - minR);
      const rad = (angleDeg * Math.PI) / 180;
      const x = centerX + radius * Math.cos(rad);
      const y = centerY + radius * Math.sin(rad);
      return `${x},${y}`;
    }).join(" ");
  };

  const computeCoords = (values) => {
    if (!values || !values.length) {
      return [
        { x: 110, y: 35 },
        { x: 170, y: 90 },
        { x: 155, y: 155 },
        { x: 110, y: 190 },
        { x: 60, y: 165 },
        { x: 45, y: 95 },
      ];
    }

    return RADAR_ANGLES_DEG.map((angleDeg, i) => {
      const scaled = normalizeForRadius(values[i]);
      const radius = minR + scaled * (maxR - minR);
      const rad = (angleDeg * Math.PI) / 180;
      const x = centerX + radius * Math.cos(rad);
      const y = centerY + radius * Math.sin(rad);
      return { x, y };
    });
  };

  // Build list of all series (base first, then overlays)
  const allSeries = [];

  allSeries.push({
    label: player || "Query player",
    values: hasBase ? baseValues : null,
    color: RADAR_SERIES_COLORS[0],
  });

  (overlaySeries || []).forEach((s, idx) => {
    const color = RADAR_SERIES_COLORS[idx + 1] || RADAR_SERIES_COLORS[1];
    allSeries.push({
      label: s.label,
      values: s.values,
      color,
    });
  });

  return (
    <div className="to-radar-card">
      <div className="to-radar-header">
        <div>
          <p className="to-radar-title">{player || "No player selected"}</p>
          <p className="to-radar-subtitle">
            {player
              ? "Radar shows percentile-normalized values for top features at this position."
              : "Pick a player to view a radar-style profile."}
          </p>
        </div>
        {pos && <span className="to-radar-pos-pill">{pos}</span>}
      </div>

      <div className="to-radar-wrapper">
        <svg viewBox="0 0 220 220" className="to-radar-svg" aria-hidden="true">
          {/* concentric circles */}
          <circle cx="110" cy="110" r="90" className="to-radar-ring" />
          <circle cx="110" cy="110" r="70" className="to-radar-ring" />
          <circle cx="110" cy="110" r="50" className="to-radar-ring" />
          <circle cx="110" cy="110" r="30" className="to-radar-ring" />
          <circle cx="110" cy="110" r="10" className="to-radar-ring" />

          {/* axes */}
          {RADAR_ANGLES_DEG.map((angle, idx) => {
            const rad = (angle * Math.PI) / 180;
            const x = centerX + 90 * Math.cos(rad);
            const y = centerY + 90 * Math.sin(rad);
            return (
              <line
                key={idx}
                x1={centerX}
                y1={centerY}
                x2={x}
                y2={y}
                className="to-radar-axis"
              />
            );
          })}

          {/* series: base + overlays */}
          {allSeries.map((series, sIdx) => {
            const pts = computePoints(series.values);
            const coords = computeCoords(series.values);
            const closingPoint = pts.split(" ")[0];
            return (
              <g key={sIdx}>
                <polygon
                  className="to-radar-polygon"
                  points={pts}
                  style={{ fill: series.color.fill }}
                />
                <polyline
                  className="to-radar-polyline"
                  points={`${pts} ${closingPoint}`}
                  style={{ stroke: series.color.stroke }}
                />
                {coords.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.x}
                    cy={p.y}
                    r="3.5"
                    className="to-radar-point"
                    style={{ fill: series.color.stroke }}
                  />
                ))}
              </g>
            );
          })}
        </svg>

        {/* metric labels around the radar */}
        <div className="to-radar-label to-radar-label-1">
          {labels?.[0] || "Metric 1"}
        </div>
        <div className="to-radar-label to-radar-label-2">
          {labels?.[1] || "Metric 2"}
        </div>
        <div className="to-radar-label to-radar-label-3">
          {labels?.[2] || "Metric 3"}
        </div>
        <div className="to-radar-label to-radar-label-4">
          {labels?.[3] || "Metric 4"}
        </div>
        <div className="to-radar-label to-radar-label-5">
          {labels?.[4] || "Metric 5"}
        </div>
        <div className="to-radar-label to-radar-label-6">
          {labels?.[5] || "Metric 6"}
        </div>
      </div>

      {/* Legend for query vs overlays */}
      <div className="to-radar-legend">
        {allSeries.map((s, idx) => (
          <div key={idx} className="to-radar-legend-item">
            <span
              className="to-radar-legend-swatch"
              style={{ backgroundColor: s.color.stroke }}
            />
            <span className="to-radar-legend-label">
              {idx === 0 ? "Query: " : "Compare: "}
              {s.label}
            </span>
          </div>
        ))}
      </div>

      {/* Feature explanation legend for this position */}
      <div className="to-radar-feature-legend">
        {labels?.map((shortKey, idx) => (
          <div key={idx} className="to-radar-feature-legend-row">
            <span className="to-radar-feature-legend-key">{shortKey}</span>
            <span className="to-radar-feature-legend-desc">
              {featureLegend && featureLegend[shortKey]
                ? featureLegend[shortKey]
                : ""}
            </span>
          </div>
        ))}
      </div>

      <p className="to-radar-footnote">
        Values are percentile-normalized within this position group for the
        selected features (higher = closer to the top performers at this
        position).
      </p>
    </div>
  );
};

function App() {
  const [selectedZone, setSelectedZone] = useState("FW");
  const [playerInput, setPlayerInput] = useState("");
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [recommendation, setRecommendation] = useState(null);
  const [overlayPlayers, setOverlayPlayers] = useState([]); // [{name, radarFeatures}]
  const [error, setError] = useState("");
  const [isRunning, setIsRunning] = useState(false);

  const currentPlayers = selectedZone
    ? PLAYER_LIST_BY_POS[selectedZone] || []
    : [];

  const filteredSuggestions =
    playerInput.trim().length === 0
      ? currentPlayers
      : currentPlayers.filter((p) =>
          p.toLowerCase().includes(playerInput.toLowerCase())
        );

  const handleZoneClick = (zoneCode) => {
    setSelectedZone(zoneCode);
    setPlayerInput("");
    setSuggestionsOpen(false);
    setRecommendation(null);
    setOverlayPlayers([]);
    setError("");
  };

  const handlePlayerChange = (e) => {
    setPlayerInput(e.target.value);
    setSuggestionsOpen(true);
    setError("");
  };

  const handleSuggestionClick = (name) => {
    setPlayerInput(name);
    setSuggestionsOpen(false);
    setError("");
  };

  const handleRunModel = () => {
    setError("");
    setRecommendation(null);
    setOverlayPlayers([]);

    if (!selectedZone) {
      setError("Please select a position filter.");
      return;
    }
    if (!playerInput.trim()) {
      setError("Please enter or select a player name.");
      return;
    }

    const name = playerInput.trim();
    const recMap = RECS_BY_POS[selectedZone] || {};

    if (!Object.prototype.hasOwnProperty.call(recMap, name)) {
      setError("That player is not in the precomputed recommendation map.");
      return;
    }

    setIsRunning(true);

    const data = recMap[name];

    setRecommendation({
      queryPlayer: name,
      neighbors: data.neighbors || [],
      evalStats: data.eval_stats || null,
      radarFeatures: data.radar_features || null,
    });

    setIsRunning(false);
  };

  const currentPosLabel =
    POSITION_ZONES.find((z) => z.code === selectedZone)?.label ?? "";

  const radarLabels = RADAR_FEATURES_BY_POS[selectedZone] || [];
  const featureLegend = RADAR_FEATURE_LEGENDS[selectedZone] || {};

  const baseRadarValues =
    recommendation?.radarFeatures && radarLabels.length === 6
      ? radarLabels.map((feat) => {
          const entry = recommendation.radarFeatures[feat];
          return entry && typeof entry.norm === "number" ? entry.norm : null;
        })
      : null;

  // Build overlay series from overlayPlayers state
  const overlaySeries =
    overlayPlayers && radarLabels.length === 6
      ? overlayPlayers.map((p) => ({
          label: p.name,
          values: radarLabels.map((feat) => {
            const entry = p.radarFeatures[feat];
            return entry && typeof entry.norm === "number" ? entry.norm : null;
          }),
        }))
      : [];

  // Toggle overlay player when clicking on a row
  const handleRowClick = (row) => {
    const fullMap = RECS_BY_POS[selectedZone] || {};
    const data = fullMap[row.Player];
    if (!data || !data.radar_features) {
      return;
    }

    setOverlayPlayers((prev) => {
      const already = prev.find((p) => p.name === row.Player);
      if (already) {
        // remove if already selected
        return prev.filter((p) => p.name !== row.Player);
      }
      // limit overlays to 3 for sanity
      const next = [...prev, { name: row.Player, radarFeatures: data.radar_features }];
      if (next.length > 3) {
        next.shift();
      }
      return next;
    });
  };

  const isRowSelected = (playerName) =>
    overlayPlayers.some((p) => p.name === playerName);

  return (
    <div className="to-root">
      <div className="to-main-card">
        <header className="to-header">
          <h1 className="to-title">Transfer Oracle</h1>
          <p className="to-subtitle">
            Select a position, search for a player, and explore similar options
            with a radar-style profile and recommendation list.
          </p>
        </header>

        {/* Top controls: position filter + search bar */}
        <div className="to-controls-row">
          <div className="to-pos-toggle">
            {POSITION_ZONES.map((zone) => (
              <button
                key={zone.code}
                className={
                  "to-pos-toggle-btn" +
                  (selectedZone === zone.code
                    ? " to-pos-toggle-btn--active"
                    : "")
                }
                onClick={() => handleZoneClick(zone.code)}
              >
                {zone.label}
              </button>
            ))}
          </div>

          <div className="to-search-block">
            <p className="to-panel-label">Player name</p>
            <div className="to-input-wrapper">
              <input
                type="text"
                className="to-input"
                placeholder={
                  selectedZone
                    ? `Start typing a ${currentPosLabel
                        .toLowerCase()
                        .slice(0, -1)}...`
                    : "Select a position first"
                }
                value={playerInput}
                onChange={handlePlayerChange}
                disabled={!selectedZone}
                onFocus={() => selectedZone && setSuggestionsOpen(true)}
              />
              <button
                className="to-run-btn"
                onClick={handleRunModel}
                disabled={isRunning}
              >
                {isRunning ? "Running..." : "Find similar"}
              </button>
            </div>

            {suggestionsOpen &&
              selectedZone &&
              filteredSuggestions.length > 0 && (
                <div className="to-suggestions">
                  {filteredSuggestions.map((name) => (
                    <button
                      key={name}
                      className="to-suggestion-item"
                      onClick={() => handleSuggestionClick(name)}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}

            {selectedZone && filteredSuggestions.length === 0 && (
              <p className="to-muted">No matches in this dataset.</p>
            )}

            {error && <p className="to-error">{error}</p>}
          </div>
        </div>

        {/* Main content: radar + table */}
        <div className="to-content-row">
          <RadarCard
            player={recommendation?.queryPlayer}
            pos={
              POSITION_ZONES.find((z) => z.code === selectedZone)?.short ?? ""
            }
            labels={radarLabels}
            baseValues={baseRadarValues}
            overlaySeries={overlaySeries}
            featureLegend={featureLegend}
          />

          <div className="to-recs-card">
            <p className="to-panel-label">Recommendations</p>

            {!recommendation && (
              <p className="to-muted">
                Choose a position, pick a player, then click{" "}
                <strong>Find similar</strong> to see precomputed
                recommendations from your clustering + KNN pipeline.
              </p>
            )}

            {recommendation && (
              <>
                <div className="to-results-header">
                  <p className="to-results-query">
                    Query player:{" "}
                    <span className="to-tag">
                      {recommendation.queryPlayer}
                    </span>
                  </p>
                  <p className="to-results-meta">
                    Position:{" "}
                    <span className="to-tag">
                      {
                        POSITION_ZONES.find(
                          (z) => z.code === selectedZone
                        )?.short
                      }
                    </span>
                  </p>
                </div>

                <div className="to-table-wrapper">
                  <table className="to-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Player</th>
                        <th>Pos</th>
                        <th>Squad</th>
                        <th>Comp</th>
                        <th>Distance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recommendation.neighbors.map((row, idx) => (
                        <tr
                          key={idx}
                          onClick={() => handleRowClick(row)}
                          style={{
                            cursor: "pointer",
                            outline: isRowSelected(row.Player)
                              ? "1px solid #22c55e"
                              : "none",
                          }}
                        >
                          <td>{idx + 1}</td>
                          <td>{row.Player}</td>
                          <td>{row.Pos}</td>
                          <td>{row.Squad}</td>
                          <td>{row.Comp}</td>
                          <td>
                            {typeof row.distance === "number"
                              ? row.distance.toFixed(3)
                              : row.distance}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        </div>

        <p className="to-footer">
          © 2025 Transfer Oracle. Data sourced from public football statistics kaggle datasets.
        </p>
      </div>
    </div>
  );
}

export default App;
