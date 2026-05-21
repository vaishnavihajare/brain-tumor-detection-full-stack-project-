async function predict() {

    let fileInput = document.getElementById("imageInput");

    if (fileInput.files.length === 0) {
        alert("Please select an image");
        return;
    }

    let formData = new FormData();

    formData.append("file", fileInput.files[0]);

    try {

        let response = await fetch("http://127.0.0.1:8000/predict", {

            method: "POST",
            body: formData

        });

        let data = await response.json();

        document.getElementById("result").innerHTML = `
            Prediction: ${data.prediction} <br>
            Confidence: ${data.confidence}
        `;

    } catch (error) {

        console.log(error);

        alert("Error connecting to API");

    }
}