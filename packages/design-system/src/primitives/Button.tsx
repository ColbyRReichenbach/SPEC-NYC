import type { ButtonHTMLAttributes } from "react";

export function Button(props: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      style={{
        borderRadius: 12,
        border: "1px solid #D8E0DB",
        background: "#0E8A6A",
        color: "#FFFFFF",
        padding: "10px 14px",
        cursor: "pointer"
      }}
    />
  );
}
