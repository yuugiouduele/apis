

export const portfolio = async (deblock: Block,) => {
    try {
        deblock.query.filter(async () => {
            await fetch("http://" + deblock.env + "/" + deblock.query[1]).then{

            }
        })
    } catch (e) {
        console.error(e)
    }
}