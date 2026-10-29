import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchPilot AI",
  description: "Evidence-backed scientific paper discovery and analysis",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
