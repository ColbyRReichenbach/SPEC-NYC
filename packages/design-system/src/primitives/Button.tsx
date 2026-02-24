import type { ButtonHTMLAttributes } from "react";

export function Button(props: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      style={{
        borderRadius: 12,
        border: "1px solid transparent",
        background: "linear-gradient(135deg, #00AE88, #283891)",
        color: "#FFFFFF",
        padding: "10px 14px",
        cursor: "pointer",
        fontWeight: 600
      }}
    />
  );
}
