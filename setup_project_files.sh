#!/bin/bash

# Create requirements.txt file for the project
cat > /home/ubuntu/stock_analysis_project/requirements.txt << 'EOL'
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
yfinance>=0.1.70
sqlalchemy>=1.4.0
plotly>=5.5.0
tabulate>=0.8.0
jupyter>=1.0.0
EOL

# Create a simple script to zip the project for easy download
cat > /home/ubuntu/stock_analysis_project/package_project.sh << 'EOL'
#!/bin/bash
cd /home/ubuntu
zip -r stock_analysis_project.zip stock_analysis_project
echo "Project packaged successfully at /home/ubuntu/stock_analysis_project.zip"
EOL

# Make the script executable
chmod +x /home/ubuntu/stock_analysis_project/package_project.sh

echo "Created requirements.txt and packaging script"
