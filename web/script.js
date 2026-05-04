document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const trainRatio = document.getElementById('train-ratio');
    const ratioVal = document.getElementById('ratio-val');
    const nComponents = document.getElementById('n-components');
    const accuracyVal = document.getElementById('accuracy-val');
    const splitVal = document.getElementById('split-val');
    const plotContainer = document.getElementById('plot-container');
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const predictResult = document.getElementById('predict-result');
    const identityVal = document.getElementById('identity-val');
    const includeRoc = document.getElementById('include-roc');

    // Update ratio display
    trainRatio.addEventListener('input', (e) => {
        ratioVal.textContent = e.target.value;
    });

    // Run Analysis
    runBtn.addEventListener('click', async () => {
        runBtn.disabled = true;
        runBtn.textContent = 'Analyzing...';
        
        try {
            const response = await fetch(`/run?n_components=${nComponents.value}&train_ratio=${trainRatio.value}&include_roc=${includeRoc.checked}`);
            const data = await response.json();
            
            if (data.accuracy !== undefined) {
                accuracyVal.textContent = (data.accuracy * 100).toFixed(1) + '%';
                splitVal.textContent = `${data.train_size} / ${data.test_size}`;
                
                // Refresh plot only if included
                if (includeRoc.checked) {
                    plotContainer.innerHTML = `<img src="/roc-plot?t=${Date.now()}" alt="ROC Curve">`;
                    plotContainer.classList.remove('empty');
                } else {
                    plotContainer.innerHTML = `<p>ROC Analysis skipped for speed</p>`;
                    plotContainer.classList.add('empty');
                }
            }
        } catch (error) {
            console.error('Error running analysis:', error);
            alert('Failed to run analysis. Check console for details.');
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Initialize & Run Analysis';
        }
    });

    // File Upload handling
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.background = 'rgba(79, 70, 229, 0.1)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        predictResult.classList.add('hidden');
        dropZone.innerHTML = '<p>Processing Image...</p>';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                identityVal.textContent = "Processing Error";
                identityVal.style.color = "#f43f5e";
                predictResult.classList.remove('hidden');
                return;
            }

            const data = await response.json();
            
            if (data.predicted_label !== undefined) {
                if (data.predicted_label === -1) {
                    identityVal.textContent = "No Face Detected";
                    identityVal.style.color = "var(--text-muted)";
                } else if (data.predicted_label === -2) {
                    identityVal.textContent = "Unknown Subject";
                    identityVal.style.color = "#f43f5e"; 
                } else {
                    identityVal.textContent = `Subject #${data.predicted_label}`;
                    identityVal.style.color = "var(--accent)";
                }
                predictResult.classList.remove('hidden');
            }
        } catch (error) {
            console.error('Error predicting:', error);
            identityVal.textContent = "Connection Error";
            identityVal.style.color = "#f43f5e";
            predictResult.classList.remove('hidden');
        } finally {
            dropZone.innerHTML = '<p>Drag & Drop an image or click to upload</p>';
        }
    }
});
