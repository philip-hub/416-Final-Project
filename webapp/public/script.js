async function generate_model() {
  let prompt = document.querySelector("#prompt").value;

  document.querySelector("#output").textContent = "generating model...";

  let response = await fetch("/generate_model", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      "prompt": prompt,
    }),
  });
  let responseJson = await response.json();

  document.querySelector("#output").textContent = `Output (took ${responseJson.generationTime}):`;

  modelJsonDiv = document.querySelector("#model-json-div");
  modelJsonDiv.innerHTML = "";

  modelJsonText = document.createElement("textarea");
  modelJsonText.textContent = responseJson.model;
  modelJsonDiv.appendChild(modelJsonText);

  copyButton = document.createElement("input");
  copyButton.type = "button";
  copyButton.value = "Copy";
  copyButton.onclick = () => {
    navigator.clipboard.writeText(modelJsonText.textContent);
    copyButton.value = "Copied";
  };
  p = document.createElement("p");
  p.appendChild(copyButton);
  modelJsonDiv.appendChild(p);
}

window.addEventListener("load", () => {
    document.querySelector("#generate").addEventListener("click", generate_model);
});