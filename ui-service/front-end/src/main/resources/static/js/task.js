var codeArea = null;

$(document).ready(function () {
    if (document.getElementById('user-name') != null)
        document.getElementById('user-name').focus();
    if (document.getElementById('source-template-task') != null)
        document.getElementById('source-template-task').focus();
    codeArea = CodeMirror.fromTextArea(document.getElementById('source-template-task'), {
        lineNumbers: true,               // показывать номера строк
        matchBrackets: true,             // подсвечивать парные скобки
        mode: "text/x-java",              // стиль подсветки,
        indentUnit: 4,                    // размер табуляции
        viewportMargin: Infinity
    });
    codeArea.setOption("theme", "darcula");
    var mac = CodeMirror.keyMap.default === CodeMirror.keyMap.macDefault;
    CodeMirror.keyMap.default[(mac ? "Cmd" : "Ctrl") + "-Space"] = "autocomplete";
});

function sendExecuteTask(idUser, idTask) {
    $('#overlay').css('display', 'block');
    var infoText = $('#source-template-task-info');
    $('#card-result').css('display', 'none');
    infoText.css('display', 'none');
    var request = {};
    request.idTask = idTask;
    request.idUser = idUser;
    if (codeArea != null)
        request.sourceTask = codeArea.getValue();
    else
        request.sourceTask = $('#source-template-task');
    $.ajax({
        url: "/auth/execute_task/execute",
        contentType: "application/json",
        type: "POST",
        dataType: "html",
        data: JSON.stringify(request),
        success: function (data) {
            $('#card-result').css('display', 'block');
            $('#overlay').css('display', 'none');
            infoText.css('display', 'none');
            $('#result-test').html($(data).find('#result-test').html());
        },
        error: function (request, status, error) {
            $('#overlay').css('display', 'none');
            var errorResponse = JSON.parse(request.responseText);
            if (errorResponse.message.indexOf("sourceTask") !== -1) {
                infoText.text("Введите код своего решения.");
            } else if (errorResponse.message === "Test service unavailable.") {
                infoText.text("Сервер тестов недоступен, повторите позже или обратитесь к администратору.");
            } else if (errorResponse.message === "CompletedTask service unavailable.") {
                infoText.text("Сервер выполнения задач недоступен, повторите позже или обратитесь к администратору.");
            } else if (errorResponse.message === "Result service unavailable.") {
                infoText.text("Сервер результатов недоступен, повторите позже или обратитесь к администратору.");
            } else if (errorResponse.message === "Test isn't executed.") {
                infoText.text("Ошибка компиляции.");
            } else {
                infoText.text("Неизвестная ошибка, повторите позже или обратитесь к администратору.");
            }
            infoText.css('display', 'block');
        }
    });
}