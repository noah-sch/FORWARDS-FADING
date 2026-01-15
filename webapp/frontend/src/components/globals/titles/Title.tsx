type TitleProps = {
    title: string[];
}

export default function Title( {title}:TitleProps ) {
    return (
        <>
            <div className="w-full font-serif font-medium text-[56px] leading-none text-c1 flex flex-col items-center justify-start">
                {title.map((string, index) => (
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