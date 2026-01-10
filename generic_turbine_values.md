# Test Data Generation - Researched Specifications

## Power Curve Parameters

### Wind Speed Thresholds
- **Cut-in Speed**: 3.5 m/s (turbine starts generating)
- **Rated Speed**: 12.5 m/s (turbine reaches maximum power)
- **Cut-out Speed**: 25.0 m/s (turbine shuts down for safety)

### Power Output
- **Rated Power**: 5000 kW (5 MW) - typical for offshore wind turbines
- **Power Calculation**: Uses cubic relationship with wind speed (P ∝ v³)
- **Below Rated**: Power increases cubically from 0 to rated power
- **Above Rated**: Power maintained at rated level via pitch control

## Rotor Speed

### Operating Ranges
- **Cut-in**: ~11 RPM (at 3.5 m/s wind speed)
- **Below Rated**: Increases from 11 RPM to ~16.4 RPM as wind increases
- **Rated**: Constant 16 RPM (maintained above rated wind speed)
- **Maximum**: ~18.5 RPM (safety limit)

### Relationship to Wind Speed
- Rotor speed increases linearly with wind speed up to rated speed
- Above rated speed, rotor speed is held constant via pitch control

## Blade Pitch Angle

### Operating Ranges
- **Below Rated Wind Speed (<12.5 m/s)**: -1° to 2° (near zero for maximum efficiency)
- **At Rated Speed (12.5 m/s)**: ~0° (optimal angle)
- **Above Rated Speed**: Increases from 0° to 27° (feathering to limit power)
- **Cut-out/Shutdown**: 85-90° (fully feathered for safety)

### Control Logic
- Below rated: Pitch optimized for maximum power capture
- Above rated: Pitch increases to reduce aerodynamic efficiency and maintain constant power
- At cut-out: Blades fully feathered to stop rotation

## Temperature Ranges

### Generator Temperature
- **Normal Operating**: 55-80°C (base 55°C + load-dependent increase)
- **Warning Threshold**: >80°C
- **Critical Threshold**: >90°C
- **Maximum Safe**: 100°C
- **Relationship**: Increases with power output (more load = more heat)

### Gearbox Oil Temperature
- **Normal Operating**: 50-70°C (base 50°C + load-dependent increase)
- **Warning Threshold**: >70°C
- **Critical Threshold**: >80°C
- **Maximum Safe**: 90°C
- **Relationship**: Increases with power output and rotor speed

### Status-Based Variations
- **Green Status**: Normal temperatures with small variations (±2-3°C)
- **Yellow Status**: Elevated temperatures (+8-10°C above normal)
- **Red Status**: Significantly elevated temperatures (+20°C above normal)

## Vibration Levels

### Operating Ranges (m/s²)
- **Normal**: 1-3 m/s² (healthy operation)
- **Warning**: 3-6 m/s² (elevated, requires attention)
- **Critical**: >6 m/s² (high risk, immediate inspection needed)
- **Maximum Acceptable**: ~15 m/s² (before emergency shutdown)

### Status-Based Variations
- **Green Status**: 1.5 m/s² average (σ=0.4)
- **Yellow Status**: 4.5 m/s² average (σ=1.2)
- **Red Status**: 8.0 m/s² average (σ=2.0)

## Gearbox Oil Pressure

### Operating Ranges (bar)
- **Normal**: 3.0-4.0 bar
- **Optimal**: 3.5 bar
- **Warning**: <3.0 bar
- **Critical**: <2.5 bar
- **Minimum Safe**: 2.0 bar
- **Maximum**: 4.5 bar

### Status-Based Variations
- **Green Status**: 3.5 bar average (σ=0.15)
- **Yellow Status**: 3.0 bar average (σ=0.25)
- **Red Status**: 2.5 bar average (σ=0.4)

## Yaw System

### Yaw Position
- **Range**: 0-360° (continuous rotation)
- **Change Rate**: Gradual, ~1.5° per reading (smooth tracking)
- **Behavior**: Follows wind direction with small lag

### Yaw Error (Misalignment)
- **Acceptable**: <5° (normal operation)
- **Warning**: 5-10° (reduced efficiency)
- **Critical**: >10° (significant power loss)
- **Maximum**: ±15° (before corrective action)
- **Power Loss**: Approximately cos²(yaw_error) relationship

## Weather Conditions

### Wind Speed
- **Typical Range**: 5-25 m/s (offshore)
- **Most Common**: 8-15 m/s
- **Variability**: Gradual changes (Gaussian with σ=1.5 m/s)
- **Correlation**: Changes smoothly over time (not random jumps)

### Wind Direction
- **Range**: 0-360°
- **Variability**: Gradual changes (Gaussian with σ=12°)
- **Correlation**: Changes smoothly, following weather patterns

### Wave Height
- **Range**: 0.1-8.0 m
- **Relationship**: Correlates with wind speed (higher wind = higher waves)
- **Formula**: ~0.3 + (wind_speed/25) × 4.0 m
- **Typical**: 1-4 m for normal conditions

### Wave Period
- **Range**: 2.5-18 seconds
- **Relationship**: Correlates with wave height
- **Formula**: ~3.5 + (wave_height/5) × 10 seconds

### Water Temperature (Offshore)
- **Range**: 5-19°C (Irish Sea / North Atlantic)
- **Typical**: 11°C average (σ=2.5°C)
- **Seasonal Variation**: Included in generation

### Air Temperature
- **Range**: -5°C to 25°C (offshore, can be colder)
- **Relationship**: Slightly higher than water temperature
- **Typical**: Water temp + 1.5°C (σ=2.5°C)

### Air Humidity
- **Range**: 50-100%
- **Typical**: 82% average (σ=10%)
- **Offshore**: Generally high due to sea proximity

### Lightning Strikes
- **Probability**: ~2% chance per reading
- **Values**: 0, 1, or 2 strikes
- **Rare Event**: Simulates actual lightning detection

## Health Prediction Calculations

### Health Score Factors
- **Vibration >3.0 m/s²**: -0.08 per m/s² above threshold
- **Gearbox Temp >65°C**: -0.025 per °C above threshold
- **Generator Temp >75°C**: -0.025 per °C above threshold
- **Oil Pressure <3.2 bar**: -0.12 per bar below threshold
- **Yaw Error >5°**: -0.015 per degree above threshold

### Failure Probability Factors
- **Vibration >3.0 m/s²**: +0.04 per m/s² above threshold
- **Gearbox Temp >65°C**: +0.025 per °C above threshold
- **Generator Temp >75°C**: +0.025 per °C above threshold
- **Oil Pressure <3.2 bar**: +0.06 per bar below threshold
- **Yaw Error >5°**: +0.01 per degree above threshold

### Status-Based Ranges
- **Green**: Health 0.75-1.0, Failure 0.0-0.15
- **Yellow**: Health 0.45-0.75, Failure 0.15-0.45
- **Red**: Health 0.15-0.55, Failure 0.4-0.9

## Alert Generation Thresholds

### Vibration Alerts
- **Warning**: >3.0 m/s²
- **Critical**: >6.0 m/s²

### Temperature Alerts
- **Gearbox Warning**: >70°C
- **Gearbox Critical**: >80°C
- **Generator Warning**: >80°C
- **Generator Critical**: >90°C

### Pressure Alerts
- **Warning**: <3.0 bar
- **Critical**: <2.5 bar

### Yaw Alerts
- **Warning**: >5° misalignment
- **Critical**: >10° misalignment

### Predictive Alerts
- **Critical**: Failure probability >50%

## Data Generation Intervals

### Timestamp Spacing
- **Interval**: 10 minutes (standard SCADA logging frequency)
- **Period**: 30 days of historical data
- **Total Readings**: 4320 per turbine (30 days × 24 hours × 6 readings/hour)

### Regularity
- All logs generated at exact 10-minute intervals
- No random gaps (unless simulating faults)
- Sequential timestamps from 30 days ago to present

## Power Output Relationships

### Wind Speed to Power
- **Below Cut-in**: 0 kW
- **Cut-in to Rated**: P = P_rated × ((v - v_cutin) / (v_rated - v_cutin))³
- **Rated to Cut-out**: P = P_rated (constant)
- **Above Cut-out**: 0 kW

### Pitch Angle Effect
- **Below Rated**: Pitch near 0° maximizes power
- **Above Rated**: Pitch increases to limit power (feathering)
- **Power Loss**: Approximately proportional to pitch angle increase

### Yaw Error Effect
- **Power Loss**: P_actual = P_ideal × cos²(yaw_error)
- **Example**: 10° yaw error ≈ 3% power loss
- **Example**: 15° yaw error ≈ 7% power loss

## Temperature Relationships

### Load-Dependent Heating
- **Generator**: Base 55°C + (power/rated_power) × 25°C
- **Gearbox**: Base 50°C + (power/rated_power) × 20°C
- **At Full Load**: Generator ~80°C, Gearbox ~70°C
- **At No Load**: Generator ~55°C, Gearbox ~50°C

### Status-Based Modifications
- **Green**: Normal operation, temperatures within safe range
- **Yellow**: Elevated temperatures indicating developing issues
- **Red**: Significantly elevated temperatures indicating serious problems

## References

### Standards & Specifications
- **IEC 61400**: International standard for wind turbine design and testing
  - IEC 61400-1: Design requirements for wind turbines
  - IEC 61400-12: Power performance measurements
  - Available at: https://webstore.iec.ch/

### Power Curve & Performance
- **Wind Turbine Power Curves**: ScienceDirect - Power Curve Engineering
  - https://www.sciencedirect.com/topics/engineering/power-curve
- **Pitch Angle Optimization**: "The Impact of the Pitch Angle on the Power of the AEOLOS V300"
  - https://www.ijrer.com/index.php/ijrer/article/download/14883/pdf
- **Pitch Control Systems**: MDPI - Wind Turbine Pitch Control
  - https://www.mdpi.com/1996-1073/17/23/5818

### Operating Parameters
- **Wind Speed Thresholds**: Energy.gov - How Do Wind Turbines Survive Severe Weather
  - https://www.energy.gov/eere/articles/how-do-wind-turbines-survive-severe-weather-and-storms
- **Rotor Speed & Control**: NREL Technical Reports
  - https://www.nrel.gov/docs/fy23osti/86519.pdf
- **Power Coefficient (Betz's Law)**: Wikipedia - Betz's Law
  - https://en.wikipedia.org/wiki/Betz%27s_law

### Temperature & Component Specifications
- **Operating Temperature Ranges**: Wind Power Engineering
  - https://www.windpowerengineering.com/operating-at-extremes-tech-note-on-turbine-operation-in-cold-climates/
- **Weather Impact Analysis**: OTS Technical Analysis
  - https://ots-tl.com/analysis-of-factors-affecting-wind-turbine-power-output/
- **Cold Climate Operations**: Natural Resources Canada
  - https://natural-resources.canada.ca/energy-sources/renewable-energy/wind-energy-cold-climates

### Yaw System & Alignment
- **Yaw Misalignment Effects**: Copernicus - Wind Energy Science
  - https://wes.copernicus.org/articles/9/385/2024/wes-9-385-2024.pdf
- **Yaw Error Power Loss**: Wikipedia - Wind Turbine Design
  - https://en.wikipedia.org/wiki/Wind_turbine_design

### Vibration & Monitoring
- **ISO 10816**: Vibration standards for rotating machinery
- **SCADA Data Patterns**: ResearchGate - Wind Turbine Monitoring
  - https://www.researchgate.net/figure/Wind-turbine-power-curve_fig1_260650465

### Weather & Environmental Data
- **Global Wind Atlas**: Wind resource data
  - https://en.wikipedia.org/wiki/Global_Wind_Atlas
- **Weather Impact on Wind Farms**: InfoPlaza
  - https://www.infoplaza.com/en/blog/the-impact-of-weather-on-wind-farms

### Academic Research
- **Power Performance Monitoring**: LEGI Grenoble
  - https://legi.grenoble-inp.fr/web/IMG/pdf/chapter_powerperformance_monitoring.pdf
- **Pitch Control Research**: Chalmers University Publications
  - https://publications.lib.chalmers.se/records/fulltext/156543.pdf

### Industry Resources
- **Wind Power Engineering**: Technical articles and calculations
  - https://www.windpowerengineering.com/
- **Energy.gov Wind Energy**: Official US Department of Energy resources
  - https://www.energy.gov/eere/wind

