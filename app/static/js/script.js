document.addEventListener('DOMContentLoaded', (event) => {
    var socket = io();
    var toggleFeature = document.getElementById('toggleFeature');

    // フォームの送信処理
    $('#chatForm').submit(function(e) {
        e.preventDefault();
        var message = $('#userInput').val().trim();
        var summaryEnabled = toggleFeature.checked;

        showLoadingScreen(); // メッセージ送信前にローディング画面を表示
        socket.emit('send_message', { message: message, summary: summaryEnabled });
        // ユーザーのメッセージに含まれる改行を<br>タグに置換して表示
        var formattedMessage = message.replace(/\n/g, '<br>');
        $('#chat-box').append(`<div class="user-message">${formattedMessage}</div>`);
        $('#userInput').val('');
        // メッセージボックスを自動スクロール
        var chatcontainer = document.getElementById('chat-container');
        chatcontainer.scrollTop = chatcontainer.scrollHeight;
    });

    socket.on('receive_message', function(data) {
        hideLoadingScreen(); // メッセージ受信後にローディング画面を非表示に
        // ボットのメッセージに含まれる改行を<br>タグに置換して表示
        var formattedMessage = data.message.replace(/\n/g, '<br>');
        $('#chat-box').append(`<div class="bot-message">${formattedMessage}</div>`);
        // メッセージボックスを自動スクロール
        var chatcontainer = document.getElementById('chat-container');
        chatcontainer.scrollTop = chatcontainer.scrollHeight;
    });

    $('#userInput').keydown(function(e) {
        if (e.ctrlKey && e.keyCode === 13) {
            $('#chatForm').submit();
            e.preventDefault();
        }
    });
});

function showLoadingScreen() {
    // ロード画面を表示するコード
    const loadingScreen = document.createElement('div');
    loadingScreen.id = 'loadingScreen';
    loadingScreen.innerHTML = '<div class="loader"></div>';
    document.body.appendChild(loadingScreen);
}

function hideLoadingScreen() {
    // ロード画面を非表示にするコード
    const loadingScreen = document.getElementById('loadingScreen');
    if (loadingScreen) {
        loadingScreen.remove();
    }
}
