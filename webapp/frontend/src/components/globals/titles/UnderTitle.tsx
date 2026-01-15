type UnderTitleProps = {
    underTitle: string[];
}

export default function UnderTitle( {underTitle}: UnderTitleProps ) {
    return (
        <>
            <div className="w-full font-sans text-sm text-c2 flex flex-col items-center justify-start">
                {underTitle.map((string, index) => (
                    <>
                        <h1 key={index}>
                            {string}
                        </h1>
                    </>
                ))}

            </div>
        </>
    )
}