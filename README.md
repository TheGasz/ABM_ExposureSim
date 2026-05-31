# ABM Waiting Room - Random Movement Simulation 

Welcome to the **Waiting Room Agent-Based Model (ABM) Simulation**! This project is an interactive, web-based simulation built with Streamlit. It allows you to design a waiting room layout and simulate the behavior of human agents as they navigate the space, look for chairs, sit down, and exit.

## ✨ Features

- * Interactive Room Design*: Use the **Design Room** tab to place obstacles, doors, and chairs on a customizable grid. You can rotate chairs and build complex layouts.
- * Agent-Based Simulation*: Agents arrive dynamically (Poisson distribution) and make decisions based on configurable probabilities (e.g., pass-through vs. sitting).
- * Ray Casting Vision*: Agents are equipped with a field of view (cone of vision). Using ray casting algorithms, the simulation calculates exactly what each agent sees.
- * Focused Agent Mode*: Isolate a single agent to observe their exact vision cone and ray hit points in real-time.
- * Obstacle Heatmap*: The simulation tracks which obstacles are observed the most and generates a dynamic heatmap. At the end of the simulation, explore a highly detailed interactive Plotly heatmap.

## 🛠️ Tech Stack

- **Python**: Core simulation logic and agent mechanics.
- **Streamlit**: Interactive web interface.
- **Matplotlib**: Real-time rendering of the grid, agents, and ray-casting.
- **Plotly**: Interactive visualization for the final obstacle heatmap.
- **NumPy**: Matrix operations and random distribution sampling.

## 🚀 How to Run

1. Make sure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open the provided local URL (usually `http://localhost:8501`) in your web browser.

## 🎮 Usage Guide

### 1. Design Room Tab
- Set the grid dimensions.
- Use the sidebar controls to add **Doors**, **Obstacles**, and **Chairs**.
- Click the grid to toggle the placement of the selected item.
- Save the layout when you are done.

### 2. Run Simulation Tab
- Configure arrival rates (λ), average sitting times, and pass-through probabilities.
- Select your **Visualization Mode**:
  - **Full Room**: View all agents and the live heatmap simultaneously.
  - **Focused Agent**: Watch a detailed, distraction-free view of a single agent's vision rays.
- Click **▶ Start** to begin!
- You can Pause, Stop, or Reset at any time. Once stopped, you can download the generated GIF.

## 📁 Project Structure

- `app.py` - Main entry point for the Streamlit application.
- `abm/` - Contains the simulation core (`model.py`) and agent logic (`agents.py`).
- `ui/` - Contains Streamlit interface components (`panel_design.py`, `panel_simulate.py`, `state.py`).
- `viz/` - Rendering scripts for the plots, ray casting, heatmap, and GIF generation.
- `constants.py` - Global configuration and color palette.
