document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("pre").forEach(function (pre) {
        const codeBlock = pre.querySelector("code");
        if (!codeBlock) return;

        const button = document.createElement("button");
        button.className = "copy-button";
        button.type = "button";
        button.title = "Copy to clipboard";
        button.innerText = "⧉";

        button.addEventListener("click", function () {
            navigator.clipboard.writeText(codeBlock.innerText).then(() => {
                button.innerText = "✓";
                setTimeout(() => button.innerText = "⧉", 2000);
            });
        });

        pre.style.position = "relative";   // key: anchor positioning
        pre.appendChild(button);           // button lives *inside* <pre>
    });
});