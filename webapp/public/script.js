async function generate_model() {
  let prompt = document.querySelector("#prompt").value;

  document.querySelector("#model").textContent = "generating model...";

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

  document.querySelector("#model").textContent = responseJson.model;
}

window.addEventListener("load", () => {
    document.querySelector("#generate").addEventListener("click", generate_model);
});