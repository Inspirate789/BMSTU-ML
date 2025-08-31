create table roles (
    id bigint generated always as identity primary key,
    name text,
    design_control boolean
);

create table tasks (
    id bigint generated always as identity primary key,
    role_id bigint references roles(id) on update cascade on delete cascade,
    name text,
    certainty boolean
);

create table dev_stages (
    id bigint generated always as identity primary key,
    name text,
    content_finalized boolean
);

create table document_formats (
    id bigint generated always as identity primary key,
    name text,
    in_use boolean,
    verification_automatizable boolean,
    certainty_of_need boolean,
    necessary boolean
);

create table verification_tools (
    id bigint generated always as identity primary key,
    name text
);

create table document_formats_and_verification_tools (
    document_format_id bigint references document_formats(id) on update cascade on delete cascade,
    verification_tool_id bigint references verification_tools(id) on update cascade on delete cascade
);

create table difficult_checks (
    id bigint generated always as identity primary key,
    name text,
    reason text
);

create table regulatory_documents (
    id bigint generated always as identity primary key,
    name text,
    versatility boolean,
    concreteness int
);
       
create table testvkr_funtions (
    id bigint generated always as identity primary key,
    name text,
    is_primary boolean,
    is_full boolean
);
