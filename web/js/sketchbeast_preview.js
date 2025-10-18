import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "sketchbeast.preview",

    async setup() {
        // Listen for preview messages from the backend
        function previewHandler(event) {
            const data = event.detail;

            // Find the Sketchbeast node that's currently executing
            const node = app.graph._nodes.find(n => n.type === "SketchbeastNode");
            if (!node) return;

            // Create or update preview image element
            const previewContainer = document.getElementById("sketchbeast-preview") || createPreviewContainer();

            // Update preview image
            const img = previewContainer.querySelector("img");
            img.src = "data:image/png;base64," + data.image;

            // Update progress text
            const progress = previewContainer.querySelector(".progress-text");
            progress.textContent = `Step ${data.step} / ${data.total}`;

            // Show container
            previewContainer.style.display = "block";
        }

        function createPreviewContainer() {
            const container = document.createElement("div");
            container.id = "sketchbeast-preview";
            container.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                background: rgba(0, 0, 0, 0.9);
                padding: 10px;
                border-radius: 8px;
                z-index: 10000;
                display: none;
            `;

            const img = document.createElement("img");
            img.style.cssText = `
                max-width: 300px;
                max-height: 300px;
                display: block;
                border-radius: 4px;
            `;

            const progress = document.createElement("div");
            progress.className = "progress-text";
            progress.style.cssText = `
                color: white;
                text-align: center;
                margin-top: 5px;
                font-family: monospace;
            `;

            container.appendChild(img);
            container.appendChild(progress);
            document.body.appendChild(container);

            return container;
        }

        // Register event listener
        app.api.addEventListener("sketchbeast.preview", previewHandler);

        // Clean up preview on queue completion
        app.api.addEventListener("executed", () => {
            const container = document.getElementById("sketchbeast-preview");
            if (container) {
                setTimeout(() => {
                    container.style.display = "none";
                }, 2000);
            }
        });
    }
});
