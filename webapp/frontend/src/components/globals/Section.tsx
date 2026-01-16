import type { ReactNode } from "react";

type SectionProps = {
    children: ReactNode;
}

export default function Section( {children}: SectionProps ) {
    return (
        <>
            <div className="w-full flex flex-col items-center justify-start py-20 gap-10">
                {children}
            </div>
        </>
    )
}