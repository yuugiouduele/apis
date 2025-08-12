git config --global http.postBuffer 1048576000

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo >> /root/.bashrc
    echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> /root/.bashrc
    eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

sudo apt-get update&&apt-get install -y build-essential

brew cleanup
brew cleanup -s
rm -rf "$(brew --cache)"
brew update
brew upgrade
brew doctor
brew install gcc
brew cleanup gcc
brew install asdf