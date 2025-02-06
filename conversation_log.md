# Conversation Log - 2025-02-05

## Initial Context
User shared LoadData.py file for analysis and potential modifications.

## Key Points Discussed:
1. File contains particle physics data analysis code
2. Code processes truth and reconstructed particle data
3. Includes momentum calculations, plotting, and optimization
4. Currently set to process 100 events (n_events = 100)

## Physics Analysis - Cos Theta Calculations

### Rest Frame Considerations
The cos theta calculations must be performed in the tau-tau rest frame for proper physics interpretation. This requires:
1. Calculating the total 4-momentum of the tau-tau system
2. Boosting all particles into this rest frame using Lorentz transformations
3. Defining the coordinate system and calculating angles in the rest frame

The boost transformation is implemented as:
```python
def boost_to_rest_frame(p, p_boost):
    beta = p_boost[1:] / p_boost[0]  # Boost vector
    beta_sq = np.dot(beta, beta)
    gamma = 1.0 / np.sqrt(1.0 - beta_sq)
    
    # Boost components
    p_parallel = np.dot(p[1:], beta) / beta_sq
    E_prime = gamma * (p[0] - np.dot(beta, p[1:]))
    p_prime = p[1:] + (gamma - 1.0) * p_parallel * beta / beta_sq - gamma * p[0] * beta
    
    return np.array([E_prime, *p_prime])
```

### Coordinate System Definition
The analysis defines a special coordinate system {r̂, n̂, k̂} in the tau-tau rest frame:
- k̂: Direction of the tau- momentum
- r̂: Perpendicular component to k̂ in the beam direction (z-axis)
- n̂: Cross product of k̂ and r̂ (normal to the decay plane)

### Angular Distributions
The code calculates three angular components for each pion:
1. cosθᵣ: Projection along r̂ axis
2. cosθₙ: Projection along n̂ axis
3. cosθₖ: Projection along k̂ axis

### Physics Significance
These cosines are used to study:
- Tau spin correlations
- Decay dynamics
- Possible CP-violating effects
- Detector performance validation

### Visualization
Results are plotted using ratio plots showing:
- Truth vs reconstructed cosθ values
- Reconstruction accuracy

## Potential Next Steps:
- Increase number of events processed
- Add additional analysis metrics
- Optimize performance
- Add error handling
- Create unit tests

## Suggested Commands:
```bash
git add conversation_log.md
git commit -m "Added physics explanation of cos theta calculations"
```
