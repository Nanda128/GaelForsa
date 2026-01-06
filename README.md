## GaelFórsa - A Predictive Maintenance Platform

GaelFórsa is an advanced predictive maintenance platform designed to help industries monitor and maintain their equipment efficiently. By leveraging cutting-edge technologies such as IoT, machine learning, and data analytics, GaelFórsa provides real-time insights into equipment health, enabling proactive maintenance strategies that reduce downtime and extend asset lifespan.

It seeks to provide a Graphics-based User Interface (GUI) for ease of use and accessibility, to manage the state of various offshore wind turbines.
It collects information from various sensors installed on the turbines, such as temperature, vibration, and pressure sensors. 

It then compares these against a list of ideal factors in order to determine the health of the turbine, and current severity.
A stretch goal is to use weather conditions to determine the ideal time to send maintenance crews.

## Quick Start

Run these commands in **two separate terminals**:

**Terminal 1 - Backend:**
```bash
cd backend && python manage.py runserver
```

**Terminal 2 - Frontend:**
```bash
cd frontend && npm install && npm run dev
```

Then open http://localhost:3000 in your browser.

**Optional commands**
```bash
python manage.py generate_test_data --turbines 10 # generate 10 turbines with 5 entries of each related model (alerts, maintaince, ect)
python manage.py delete_test_data # used to delete test entries 

****

Below is a list of the parameters we want to collect to support our system:
...TODO

****

# Members:

- Hersh Singh
- Leonards Cirksis
- Nanda Vinay
- Rachel Keaveney
- Shay Power
