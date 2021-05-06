function openInputModal() {
    var inputModal = $('#input-modal');
    inputModal.on('shown.bs.modal', function () {
        document.getElementById("user-name").focus();
    });
    inputModal.modal('show');
}

$(document).ready(function() {
    $('#btn-input').click(function () {
        var userName = document.getElementById("user-name");
        var password = document.getElementById("password");
        var infoText = $('#info-text');
        $('#overlay').css('display', 'block');
        infoText.text("Некорректные данные, неверный логин или пароль.");
        infoText.css("display", "none");
        var usernameText = userName.value;
        var passwordText = password.value;
        $.ajax({
            url: "/oauth/token",
            type: "POST",
            contentType: "application/json; charset=utf-8",
            dataType: "html",
            data: JSON.stringify({
                username: usernameText,
                password: passwordText,
                client_id: "973db60a-0b2b-44af-9f58-c3b0fe4d175f",
                client_secret: "637d813a-3f1d-47bb-bed9-f2be36ab0507",
                grant_type: "password"
            }),
            success: function (token) {
                $('#overlay').css('display', 'none');
                if (token !== null && token !== "") {
                    userName.classList.remove('is-invalid');
                    password.classList.remove('is-invalid');
                    userName.classList.add('is-valid');
                    password.classList.add('is-valid');
                    infoText.css("display", "none");
                    document.location.reload();
                } else {
                    userName.classList.remove('is-valid');
                    password.classList.remove('is-valid');
                    userName.classList.add('is-invalid');
                    password.classList.add('is-invalid');
                    infoText.css("display", "block");
                }
            },
            error: function(request) {
                $('#overlay').css('display', 'none');
                var r = JSON.parse(request.responseText);
                if (r.status === 401) {
                    userName.classList.remove('is-valid');
                    password.classList.remove('is-valid');
                    userName.classList.add('is-invalid');
                    password.classList.add('is-invalid');
                    infoText.css("display", "block");
                } else {
                    infoText.text("Сервер недоступен или возникла другая ошибка.\nОбратитесь к администратору.");
                    infoText.css("display", "block");
                }
            }
        });
        return false;
    });
    var userNameField = $('#user-name');
    var passwordField = $('#password');
    userNameField.on('input', function() {
        clearInputValid();
    });
    passwordField.on('input', function() {
        clearInputValid();
    });
    userNameField.keypress(function(e) {
        if (e.which === 13) {
            $('#btn-input').click();
        }
    });
    passwordField.keypress(function(e) {
        if (e.which === 13) {
            $('#btn-input').click();
        }
    });
});

function clearInputValid() {
    var userName = document.getElementById("user-name");
    userName.classList.remove('is-valid');
    userName.classList.remove('is-invalid');
    var password = document.getElementById("password");
    password.classList.remove('is-valid');
    password.classList.remove('is-invalid');
    $('#info-text').css("display", "none");
}

function proof_right(redirect_uri, client_id) {
    $.ajax({
        url: "/oauth/proof_right?client_id="+client_id + "&redirect_uri=" + redirect_uri,
        type: "GET",
        contentType: "application/json; charset=utf-8",
        dataType: "json",
        success: function (authorization_code) {
            alert(authorization_code);
            if (authorization_code.code !== null && authorization_code.code !== "") {
                document.location.href = redirect_uri + "?code="
                    + authorization_code.code;
            }
        },
        error: function (request, status, error) {
        }
    });
}

function cancel() {
    document.location.href = "/";
}