#!/bin/bash
echo "Running post-deployment tasks..."

# Initialize the database
python populate_database.py --reset

echo "Post-deployment tasks completed."