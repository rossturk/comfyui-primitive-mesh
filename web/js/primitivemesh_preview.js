import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "primitivemesh.preview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "PrimitiveMeshNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Create preview widget
                this.previewWidget = this.addDOMWidget("preview", "preview", document.createElement("div"));
                this.previewWidget.serializeValue = () => undefined;

                // Style the container
                const container = this.previewWidget.element;
                container.style.cssText = `
                    width: 100%;
                    min-height: 200px;
                    background: #1a1a1a;
                    border-radius: 4px;
                    padding: 10px;
                    box-sizing: border-box;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                `;

                // Create preview image
                const img = document.createElement("img");
                img.style.cssText = `
                    max-width: 100%;
                    max-height: 400px;
                    display: none;
                    border-radius: 4px;
                    margin-bottom: 10px;
                `;
                container.appendChild(img);

                // Create progress text
                const progress = document.createElement("div");
                progress.style.cssText = `
                    color: #999;
                    text-align: center;
                    font-family: monospace;
                    font-size: 12px;
                `;
                progress.textContent = "Ready";
                container.appendChild(progress);

                // Store references
                this.previewImage = img;
                this.previewProgress = progress;

                return result;
            };
        }
    },

    async setup() {
        // Listen for preview messages from the backend
        app.api.addEventListener("primitivemesh.preview", (event) => {
            const data = event.detail;

            // Find the PrimitiveMesh node that's currently executing
            const node = app.graph._nodes.find(n => n.type === "PrimitiveMeshNode");
            if (!node || !node.previewImage) return;

            // Update preview image
            node.previewImage.src = "data:image/png;base64," + data.image;
            node.previewImage.style.display = "block";

            // Update progress text
            node.previewProgress.textContent = `Step ${data.step} / ${data.total}`;
        });

        // Reset preview on queue completion
        app.api.addEventListener("executed", () => {
            const nodes = app.graph._nodes.filter(n => n.type === "PrimitiveMeshNode");
            nodes.forEach(node => {
                if (node.previewProgress) {
                    node.previewProgress.textContent = "Complete";
                }
            });
        });
    }
});
