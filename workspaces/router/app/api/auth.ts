

interface login {
    username: string,
    password: string,
}

export const Auth = async (user: login, version: number) => {
    const auth = await fetch('https://' + 'auth' + version)
        .then(res => res.json)
        .then(data => { })
}