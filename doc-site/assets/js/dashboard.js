// dashboard.js - Dynamic Dashboard Implementation
document.addEventListener("DOMContentLoaded", function() {
    console.log("Dashboard JS loaded.");
  
    // Helper: fetch and parse CSV
    function fetchCSV(url) {
      return fetch(url)
        .then(response => response.text())
        .then(text => {
          const lines = text.trim().split("\n");
          const headers = lines[0].split(",").map(h => h.trim());
          const data = lines.slice(1).map(line => {
            const cols = line.split(",").map(c => c.trim());
            let obj = {};
            headers.forEach((header, i) => {
              obj[header] = cols[i];
            });
            return obj;
          });
          return data;
        });
    }
  
    // Compute average return per model from the compiled CSV and update a bar chart
    function updateDashboardChart(data) {
      const modelReturns = {};
      data.forEach(row => {
        const model = row["Model"];
        const ret = parseFloat(row["Return [%]"]) || 0;
        if (!modelReturns[model]) {
          modelReturns[model] = { sum: 0, count: 0 };
        }
        modelReturns[model].sum += ret;
        modelReturns[model].count++;
      });
      const labels = Object.keys(modelReturns);
      const averages = labels.map(model => (modelReturns[model].sum / modelReturns[model].count).toFixed(2));
  
      const ctx = document.getElementById('dashboardChart').getContext('2d');
      if (window.dashboardChart) window.dashboardChart.destroy();
      window.dashboardChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: "Average Return (%)",
            data: averages,
            backgroundColor: "#007aff"
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false
        }
      });
    }
  
    // Load compiled backtest CSV and update chart
    document.getElementById('load-data').addEventListener('click', function() {
      fetchCSV("../results/compiled_backtest_stats.csv")
        .then(data => {
          console.log("Dashboard CSV data loaded:", data);
          updateDashboardChart(data);
        })
        .catch(err => {
          console.error("Error fetching CSV data:", err);
        });
    });
  });
  