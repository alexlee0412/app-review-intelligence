import type { Metadata } from "next";
import "./theme.css";
import styles from "./page.module.css";

export const metadata: Metadata = {
  title: "App Review Intelligence",
  description: "Ask what users are saying across app reviews.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={styles.body}>{children}</body>
    </html>
  );
}
