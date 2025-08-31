insert into roles(name, design_control)
values ('студент', true),
       ('научный руководитель', false),
       ('реценент', false),
       ('нормоконтролер', true);

insert into tasks(role_id, name, certainty)
values (1, 'писать работу', true),
       (2, 'проверять содержание', true),
       (3, 'проверять содержание', true),
       (4, 'проверять оформление', true),
       (2, 'проверять оформление', false);

insert into dev_stages(name, content_finalized)
values ('написание работы', false),
       ('проверка научным руководителем', false),
       ('рецензия', true),
       ('нормоконтроль', true);

insert into document_formats(name, in_use, verification_automatizable, certainty_of_need, necessary)
values ('листы бумаги с печатным текстом', true, false, false, true),
       ('листы бумаги с рукописным текстом', true, false, false, false),
       ('pdf', true, true, true, true),
       ('word', false, true, false, false),
       ('исходники tex', false, true, false, false),
       ('сканы', false, true, false, false),
       ('скриншоты электронной версии', false, true, false, false);

insert into verification_tools(name)
values ('глазомер'),
       ('линейка'),
       ('testvkr');

insert into document_formats_and_verification_tools(document_format_id, verification_tool_id)
values (1, 1),
       (1, 2),
       (2, 1),
       (2, 2),
       (3, 3);

insert into difficult_checks(name, reason)
values ('Количество абзацных отступов', 'Это можно делать тремя различными способами: табуляциями, выравниванием или пробелами'),
       ('Приложения', 'Их содержание очень разнообразно: от чертежей на А1 (ИУ1) до листингов кода (ИУ7); для унификаии принято оформлять их как картинки, но это не полное решение проблемы'),
       ('Заголовок "ВВЕДЕНИЕ"', 'Студенты могут принести большое множество вариантов неправильного оформления'),
       ('Размер абзацного отступа', 'Бумажная и электронная версия могут иметь разные отступы из-за ошибок при печати'),
       ('Маркировка списков', 'Списки могут быть вложенными'),
       ('Нумерация рисунков и таблиц', 'Может быть как в рамках одного раздела, так и сквозная по всему документу'),
       ('Ссылки на источники', 'Можно перепутать с обозначением отрезка'),
       ('Размер абзацного отступа', 'Бумажная и электронная версия могут иметь разные отступы из-за ошибок при печати');

insert into regulatory_documents(name, versatility, concreteness)
values ('ГОСТ', true, 0),
       ('приложения к ГОСТ для ВУЗ', false, 1),
       ('требования кафедры', false, 2);

insert into testvkr_funtions(name, is_primary, is_full)
values ('проверка плагиата', true, true),
       ('загрузка в БД антиплагиата', true, true),
       ('проверка наличия разделов', false, true),
       ('проверка корректости оформления структурных элементов', false, false);
