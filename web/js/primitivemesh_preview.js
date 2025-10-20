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
                `;
                container.appendChild(img);

                // Store reference
                this.previewImage = img;

                return result;
            };
        }
    },

    async setup() {
        // Listen for preview messages from the backend
        app.api.addEventListener("primitivemesh.preview", (event) => {
            const data = event.detail;

            console.log("Preview received for node_id:", data.node_id);
            console.log("Available nodes:", app.graph._nodes.filter(n => n.type === "PrimitiveMeshNode").map(n => ({id: n.id, type: n.type})));

            // Find the specific PrimitiveMesh node by its unique ID
            const node = app.graph._nodes.find(n => n.type === "PrimitiveMeshNode" && n.id == data.node_id);
            if (!node) {
                console.log("Node not found for id:", data.node_id);
                return;
            }
            if (!node.previewImage) {
                console.log("Node found but no previewImage:", node);
                return;
            }

            // Update preview image
            console.log("Updating preview for node:", node.id);
            node.previewImage.src = "data:image/png;base64," + data.image;
            node.previewImage.style.display = "block";
        });
    }
});
